#!/usr/bin/env python3
"""
group_duplicates.py

Finds NEAR-DUPLICATE photos — the same shot saved twice, burst frames,
screenshots of a photo, mild crops/edits/resizes — and groups them together
so you can keep the best one and ditch the rest.

It uses perceptual hashing (offline, no cloud, no AI model). Each image gets a
short fingerprint; images whose fingerprints are within a small distance of
each other are treated as the same shot.

NOTE: this groups *near-identical* shots. It does NOT group the same tank from
different angles — that needs a different (embedding) approach. Use this to
de-clutter bursts and copies.

OUTPUT (copies only — originals untouched):
    _grouped/
      group_001/        <- 2+ near-identical photos
      group_002/
      ...
      unique/           <- photos with no near-duplicate
      unreadable/       <- files that couldn't be opened

USAGE
-----
    python3 group_duplicates.py "/path/to/photos"
    python3 group_duplicates.py "/path/to/photos" --out "/path/to/result"
    python3 group_duplicates.py "/path/to/photos" --threshold 5

--threshold is how strict the match is (Hamming distance between hashes):
    0   = pixel-identical only
    1-3 = same shot, tiny differences  (safe default is 3)
    4-8 = looser; catches heavier edits but risks false groupings
Recurses into subfolders. Re-running rebuilds the _grouped folder fresh.
"""

import argparse
import os
import shutil
import sys

from PIL import Image
import imagehash

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def collect_images(src, out):
    files = []
    for root, _, names in os.walk(src):
        if os.path.abspath(root).startswith(out):
            continue  # don't re-scan our own output
        for n in names:
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, n))
    return files


def main():
    ap = argparse.ArgumentParser(description="Group near-duplicate photos.")
    ap.add_argument("folder", help="Folder of photos (searched recursively).")
    ap.add_argument("--out", help="Output folder (default: <folder>/_grouped).")
    ap.add_argument("--threshold", type=int, default=3,
                    help="Match strictness 0-8 (default 3). Lower = stricter.")
    args = ap.parse_args()

    src = os.path.abspath(args.folder)
    if not os.path.isdir(src):
        sys.exit(f"Not a folder: {src}")
    out = os.path.abspath(args.out) if args.out else os.path.join(src, "_grouped")

    # Fresh output each run so groups don't get stale.
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    files = collect_images(src, out)
    if not files:
        sys.exit(f"No images found in {src}")

    print(f"Found {len(files)} images. Hashing ...")

    # 1. Hash every image.
    hashes = []          # list of (path, hash)
    unreadable = []
    for i, path in enumerate(files, 1):
        try:
            with Image.open(path) as im:
                h = imagehash.phash(im)        # 64-bit perceptual hash
            hashes.append((path, h))
        except Exception:
            unreadable.append(path)
        if i % 50 == 0 or i == len(files):
            print(f"  hashed {i}/{len(files)}")

    # 2. Greedy grouping: each image joins the first existing group whose
    #    representative hash is within threshold; otherwise it starts a group.
    groups = []          # list of dicts: {"rep": hash, "members": [paths]}
    for path, h in hashes:
        placed = False
        for g in groups:
            if (h - g["rep"]) <= args.threshold:   # Hamming distance
                g["members"].append(path)
                placed = True
                break
        if not placed:
            groups.append({"rep": h, "members": [path]})

    # 3. Write output. Multi-member groups -> group_NNN; singletons -> unique/.
    dup_groups = [g for g in groups if len(g["members"]) > 1]
    singles = [g["members"][0] for g in groups if len(g["members"]) == 1]

    unique_dir = os.path.join(out, "unique")
    os.makedirs(unique_dir, exist_ok=True)

    def copy_in(src_path, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        name = os.path.basename(src_path)
        dest = os.path.join(dest_dir, name)
        stem, ext = os.path.splitext(name)
        n = 1
        while os.path.exists(dest):       # avoid clobbering same-named files
            dest = os.path.join(dest_dir, f"{stem}_{n}{ext}")
            n += 1
        shutil.copy2(src_path, dest)

    for idx, g in enumerate(sorted(dup_groups,
                                   key=lambda x: len(x["members"]),
                                   reverse=True), 1):
        gdir = os.path.join(out, f"group_{idx:03d}")
        for p in g["members"]:
            copy_in(p, gdir)

    for p in singles:
        copy_in(p, unique_dir)

    for p in unreadable:
        copy_in(p, os.path.join(out, "unreadable"))

    dup_count = sum(len(g["members"]) for g in dup_groups)
    print("\nDone.")
    print(f"  near-duplicate groups: {len(dup_groups)}  ({dup_count} photos)")
    print(f"  unique photos:         {len(singles)}")
    print(f"  unreadable:            {len(unreadable)}")
    print(f"\nReview the results in: {out}")
    print("Each group_NNN folder holds copies of one shot — keep your favorite,")
    print("delete the rest. Originals were NOT touched.")


if __name__ == "__main__":
    main()
