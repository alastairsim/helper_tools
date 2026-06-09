#!/usr/bin/env python3
"""
sort_photos.py

Scans a folder of photos, detects faces, and COPIES each photo into one of
three folders so you can review before committing to anything:

    likely-personal/   -> one or more faces detected (probably personal)
    likely-business/   -> no faces detected (probably a tank / equipment shot)
    review/            -> couldn't be read, or a borderline single small face

Nothing is moved or deleted. Originals are left untouched. You eyeball the
'likely-personal' and 'review' piles (much smaller than the whole set) and
drag-correct any mistakes by hand.

USAGE
-----
1. Put all the photos you want to sort in ONE folder (e.g. your _Unsorted inbox,
   synced locally via Google Drive for Desktop).
2. Run:
       python3 sort_photos.py "/path/to/your/photo/folder"
   Optionally choose where output goes:
       python3 sort_photos.py "/path/to/photos" --out "/path/to/sorted"

That's it. Re-running is safe; it skips files already sorted.
"""

import argparse
import os
import shutil
import sys
import urllib.request

import cv2

# Image extensions we'll attempt to read.
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic"}

# OpenCV DNN face detector model files (small, downloaded once).
PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Detection tuning.
CONF_STRONG = 0.60   # >= this confidence on any face => likely personal
CONF_WEAK = 0.35     # between weak and strong, or a tiny face => review pile
MIN_FACE_FRAC = 0.012  # face smaller than this fraction of image width -> uncertain


def download_model(cache_dir):
    """Fetch the two model files once into a cache directory."""
    os.makedirs(cache_dir, exist_ok=True)
    proto = os.path.join(cache_dir, "deploy.prototxt")
    model = os.path.join(cache_dir, "res10_face.caffemodel")
    for url, path in ((PROTO_URL, proto), (MODEL_URL, model)):
        if not os.path.exists(path):
            print(f"  downloading {os.path.basename(path)} ...")
            urllib.request.urlretrieve(url, path)
    return proto, model


def classify(img, net):
    """
    Return one of: 'personal', 'business', 'review'.
    'personal' = a confident face. 'review' = a weak/tiny face. 'business' = none.
    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    best = "business"
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < CONF_WEAK:
            continue
        # Face box width as a fraction of the image, to discount tiny artifacts.
        x1 = detections[0, 0, i, 3] * w
        x2 = detections[0, 0, i, 5] * w
        face_frac = abs(x2 - x1) / max(w, 1)

        if conf >= CONF_STRONG and face_frac >= MIN_FACE_FRAC:
            return "personal"  # confident, real-sized face -> done
        else:
            best = "review"    # something face-like but uncertain
    return best


def safe_dest(dest_dir, name, src_path):
    """
    Return a destination path in dest_dir for `name` that won't clobber a
    different file.

    - If nothing with that name exists there, use it as-is.
    - If a file with that name exists AND has identical size, treat it as the
      same photo already sorted -> return None (skip, makes re-runs safe).
    - Otherwise it's a real name collision from a different source folder
      -> append _1, _2, ... until the name is free.
    """
    base = os.path.join(dest_dir, name)
    if not os.path.exists(base):
        return base
    if os.path.getsize(base) == os.path.getsize(src_path):
        return None  # same-named, same-size: assume already copied, skip
    stem, ext = os.path.splitext(name)
    n = 1
    while True:
        candidate = os.path.join(dest_dir, f"{stem}_{n}{ext}")
        if not os.path.exists(candidate):
            return candidate
        if os.path.getsize(candidate) == os.path.getsize(src_path):
            return None  # this exact file already landed under a renamed copy
        n += 1


def main():
    ap = argparse.ArgumentParser(description="Sort photos by face detection.")
    ap.add_argument("folder", help="Folder containing the photos to sort.")
    ap.add_argument("--out", help="Output folder (default: <folder>/_sorted).")
    args = ap.parse_args()

    src = os.path.abspath(args.folder)
    if not os.path.isdir(src):
        sys.exit(f"Not a folder: {src}")

    out = os.path.abspath(args.out) if args.out else os.path.join(src, "_sorted")
    dirs = {
        "personal": os.path.join(out, "likely-personal"),
        "business": os.path.join(out, "likely-business"),
        "review": os.path.join(out, "review"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print("Loading face detector ...")
    proto, model = download_model(os.path.join(out, "_model"))
    net = cv2.dnn.readNetFromCaffe(proto, model)

    # Gather image files (skip our own output folder).
    files = []
    for root, _, names in os.walk(src):
        if os.path.abspath(root).startswith(out):
            continue
        for n in names:
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, n))

    if not files:
        sys.exit(f"No images found in {src}")

    print(f"Found {len(files)} images. Sorting (copying, not moving) ...\n")
    counts = {"personal": 0, "business": 0, "review": 0}

    renamed = 0
    skipped = 0

    for i, path in enumerate(files, 1):
        name = os.path.basename(path)

        img = cv2.imread(path)
        if img is None:
            label = "review"          # unreadable (e.g. HEIC without codec)
            note = "unreadable -> review"
        else:
            label = classify(img, net)
            note = label

        dest = safe_dest(dirs[label], name, path)
        if dest is None:
            skipped += 1
            print(f"[{i}/{len(files)}] {name}: already sorted, skipped")
            continue

        shutil.copy2(path, dest)
        counts[label] += 1
        if os.path.basename(dest) != name:
            renamed += 1
            print(f"[{i}/{len(files)}] {name}: {note}  (renamed -> {os.path.basename(dest)})")
        else:
            print(f"[{i}/{len(files)}] {name}: {note}")

    print("\nDone.")
    print(f"  likely-personal: {counts['personal']}")
    print(f"  likely-business: {counts['business']}")
    print(f"  review:          {counts['review']}")
    if renamed:
        print(f"  (auto-renamed to avoid name clashes: {renamed})")
    if skipped:
        print(f"  (skipped, already sorted: {skipped})")
    print(f"\nReview the results in: {out}")
    print("Originals were NOT touched. Drag-correct any mistakes by hand.")


if __name__ == "__main__":
    main()