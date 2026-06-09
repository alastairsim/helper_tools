#!/usr/bin/env python3
"""
group_by_object.py

Groups photos of the SAME object/asset taken from DIFFERENT angles — e.g. the
same truck or frac tank shot from front, side, and rear. This is the harder
cousin of near-duplicate detection: it works even when the pixels differ a lot.

How it works (all offline, no cloud):
  1. Each photo is run through a small pretrained vision network (SqueezeNet)
     to produce an "embedding" — a vector that captures what's in the image.
  2. Photos whose embeddings are close together (cosine similarity above a
     threshold) get clustered into the same group.

The model (~5 MB) downloads once from GitHub on first run.

IMPORTANT — expectation setting:
  This groups by overall visual content. It reliably pulls together multiple
  shots of the same asset, but it is a SUGGESTION engine, not ground truth:
    - Two different but similar-looking white trucks may land in one group.
    - The same asset shot in very different settings may split.
  Treat the groups as a strong first pass, then eyeball and correct.

OUTPUT (copies only — originals untouched):
    _objects/
      group_001/   <- photos that look like the same asset
      group_002/
      ...
      (singletons also get their own group_NNN)
      unreadable/  <- files that couldn't be opened

USAGE
-----
    python3 group_by_object.py "/path/to/photos"
    python3 group_by_object.py "/path/to/photos" --out "/path/to/result"
    python3 group_by_object.py "/path/to/photos" --threshold 0.90

--threshold is the similarity cutoff (0-1). Higher = stricter (fewer photos per
group, less chance of mixing different assets). Lower = looser (bigger groups,
more risk of merging look-alikes). Default 0.92 worked well in testing; if
different assets are getting merged, raise it toward 0.95; if obvious matches
are being split, lower it toward 0.88.

Recurses into subfolders. Re-running rebuilds the _objects folder fresh.
"""

import argparse
import os
import shutil
import sys
import urllib.request

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Small classification net committed directly to GitHub (no Git-LFS), so it
# downloads reliably. We use an intermediate layer as a feature extractor.
PROTO_URL = "https://raw.githubusercontent.com/forresti/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/forresti/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel"
FEATURE_LAYER = "pool10"   # global pooling layer -> 1000-d descriptor


def get_model(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    proto = os.path.join(cache_dir, "squeezenet.prototxt")
    model = os.path.join(cache_dir, "squeezenet.caffemodel")
    for url, path in ((PROTO_URL, proto), (MODEL_URL, model)):
        if not os.path.exists(path):
            print(f"  downloading {os.path.basename(path)} (one time) ...")
            urllib.request.urlretrieve(url, path)
    return proto, model


def embed(net, img):
    """Return an L2-normalized feature vector for one image (BGR ndarray)."""
    blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227),
                                 (104, 117, 123), swapRB=False, crop=True)
    net.setInput(blob)
    v = net.forward(FEATURE_LAYER).flatten().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def collect_images(src, out):
    files = []
    for root, _, names in os.walk(src):
        if os.path.abspath(root).startswith(out):
            continue
        for n in names:
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, n))
    return files


def copy_in(src_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    name = os.path.basename(src_path)
    dest = os.path.join(dest_dir, name)
    stem, ext = os.path.splitext(name)
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(dest_dir, f"{stem}_{n}{ext}")
        n += 1
    shutil.copy2(src_path, dest)


def main():
    ap = argparse.ArgumentParser(description="Group photos of the same object.")
    ap.add_argument("folder", help="Folder of photos (searched recursively).")
    ap.add_argument("--out", help="Output folder (default: <folder>/_objects).")
    ap.add_argument("--threshold", type=float, default=0.92,
                    help="Similarity cutoff 0-1 (default 0.92). Higher=stricter.")
    args = ap.parse_args()

    src = os.path.abspath(args.folder)
    if not os.path.isdir(src):
        sys.exit(f"Not a folder: {src}")
    out = os.path.abspath(args.out) if args.out else os.path.join(src, "_objects")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    print("Loading model ...")
    proto, model = get_model(os.path.join(out, "_model"))
    net = cv2.dnn.readNetFromCaffe(proto, model)

    files = collect_images(src, out)
    if not files:
        sys.exit(f"No images found in {src}")

    print(f"Found {len(files)} images. Computing visual fingerprints ...")
    vectors = []      # (path, embedding)
    unreadable = []
    for i, path in enumerate(files, 1):
        img = cv2.imread(path)
        if img is None:
            unreadable.append(path)
        else:
            vectors.append((path, embed(net, img)))
        if i % 25 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)}")

    # Greedy clustering by cosine similarity to each group's running centroid.
    # Simple, order-independent enough for this use, and needs no extra deps.
    groups = []   # list of {"centroid": vec, "members": [paths], "vecs": [..]}
    for path, v in vectors:
        best_g, best_sim = None, -1.0
        for g in groups:
            sim = float(np.dot(v, g["centroid"]))
            if sim > best_sim:
                best_sim, best_g = sim, g
        if best_g is not None and best_sim >= args.threshold:
            best_g["members"].append(path)
            best_g["vecs"].append(v)
            c = np.mean(best_g["vecs"], axis=0)
            best_g["centroid"] = c / (np.linalg.norm(c) + 1e-9)
        else:
            groups.append({"centroid": v, "members": [path], "vecs": [v]})

    # Write output: largest groups first so group_001 is the biggest cluster.
    groups.sort(key=lambda g: len(g["members"]), reverse=True)
    multi = sum(1 for g in groups if len(g["members"]) > 1)
    for idx, g in enumerate(groups, 1):
        gdir = os.path.join(out, f"group_{idx:03d}")
        for p in g["members"]:
            copy_in(p, gdir)
    for p in unreadable:
        copy_in(p, os.path.join(out, "unreadable"))

    print("\nDone.")
    print(f"  total groups:        {len(groups)}")
    print(f"  multi-photo groups:  {multi}")
    print(f"  single-photo groups: {len(groups) - multi}")
    print(f"  unreadable:          {len(unreadable)}")
    print(f"\nReview the results in: {out}")
    print("Each group_NNN folder is one suspected asset. Merge/split by hand as")
    print("needed — this is a strong first pass, not the final word. Originals")
    print("were NOT touched.")


if __name__ == "__main__":
    main()