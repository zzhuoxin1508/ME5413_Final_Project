#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Render occupancy map from ROS map yaml/pgm.")
    parser.add_argument("--map-yaml", required=True, help="Path to map yaml (e.g. my_map.yaml)")
    parser.add_argument("--out", default="", help="Output png path")
    args = parser.parse_args()

    map_yaml = Path(args.map_yaml).resolve()
    with open(map_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    image_path = (map_yaml.parent / cfg["image"]).resolve()
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    occupied_thresh = float(cfg.get("occupied_thresh", 0.65))
    free_thresh = float(cfg.get("free_thresh", 0.196))
    negate = int(cfg.get("negate", 0))

    occ_t = int(round(occupied_thresh * 255.0))
    free_t = int(round(free_thresh * 255.0))

    if negate == 0:
        occ = arr <= occ_t
        free = arr >= free_t
    else:
        occ = arr >= (255 - occ_t)
        free = arr <= (255 - free_t)

    unknown = ~(occ | free)

    # black=occupied, white=free, gray=unknown
    out = np.full(arr.shape, 127, dtype=np.uint8)
    out[free] = 255
    out[occ] = 0

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = map_yaml.with_name(map_yaml.stem + "_occupancy_preview.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(out_path)

    print(f"Saved preview: {out_path}")
    print(f"map image: {image_path}")
    print(f"size: {arr.shape[1]}x{arr.shape[0]}")
    print(f"occupied pixels: {int(np.sum(occ))}, free pixels: {int(np.sum(free))}, unknown pixels: {int(np.sum(unknown))}")


if __name__ == "__main__":
    main()
