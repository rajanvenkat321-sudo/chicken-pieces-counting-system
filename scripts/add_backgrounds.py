"""
add_backgrounds.py
------------------
Downloads and integrates non-chicken food images (burgers, fries, pizza, empty plates)
into the dataset as negative background samples to prevent false positive detections.

Uses Unsplash's high-speed CDN with automatic 640px resizing to minimize file sizes.

Usage:
    python scripts/add_backgrounds.py
"""

import sys
import time
import urllib.request
import ssl
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR   = PROJECT_ROOT / "datasets" / "images"
LABELS_DIR   = PROJECT_ROOT / "datasets" / "labels"

# High-quality fast-food images from Unsplash CDN resized to 640px (YOLO's expected size)
# These contain NO chicken, so they are perfect negative samples.
BACKGROUND_SOURCES = [
    # Train set background samples (11 images)
    ("https://images.unsplash.com/photo-1568901346375-23c9450c58cd?q=80&w=640&auto=format&fit=crop", "bg_cheeseburger.jpg", "train"),
    ("https://images.unsplash.com/photo-1561758033-d89a9ad46330?q=80&w=640&auto=format&fit=crop", "bg_burger_fries.jpg", "train"),
    ("https://images.unsplash.com/photo-1573080496219-bb080dd4f877?q=80&w=640&auto=format&fit=crop", "bg_french_fries.jpg", "train"),
    ("https://images.unsplash.com/photo-1513104890138-7c749659a591?q=80&w=640&auto=format&fit=crop", "bg_pizza_margherita.jpg", "train"),
    ("https://images.unsplash.com/photo-1590947132387-155cc02f3212?q=80&w=640&auto=format&fit=crop", "bg_pizza_slice.jpg", "train"),
    ("https://images.unsplash.com/photo-1619740455993-9e612b1af08a?q=80&w=640&auto=format&fit=crop", "bg_hot_dog.jpg", "train"),
    ("https://images.unsplash.com/photo-1588720813957-658b7842730f?q=80&w=640&auto=format&fit=crop", "bg_empty_plate.jpg", "train"),
    ("https://images.unsplash.com/photo-1596797038530-2c107229654b?q=80&w=640&auto=format&fit=crop", "bg_wooden_board.jpg", "train"),
    ("https://images.unsplash.com/photo-1512621776951-a57141f2eefd?q=80&w=640&auto=format&fit=crop", "bg_salad.jpg", "train"),
    ("https://images.unsplash.com/photo-1551024601-bec78aea704b?q=80&w=640&auto=format&fit=crop", "bg_donuts.jpg", "train"),
    ("https://images.unsplash.com/photo-1482049016688-2d3e1b311543?q=80&w=640&auto=format&fit=crop", "bg_sandwich.jpg", "train"),

    # Val set background samples (4 images)
    ("https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?q=80&w=640&auto=format&fit=crop", "bg_pancakes.jpg", "val"),
    ("https://images.unsplash.com/photo-1547592180-85f173990554?q=80&w=640&auto=format&fit=crop", "bg_soup.jpg", "val"),
    ("https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?q=80&w=640&auto=format&fit=crop", "bg_pizza_val.jpg", "val"),
    ("https://images.unsplash.com/photo-1540189549336-e6e99c3679fe?q=80&w=640&auto=format&fit=crop", "bg_salad_val.jpg", "val"),
]


def download_image(url: str, dest_path: Path):
    """Download an image with a browser User-Agent."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    # Ignore SSL verification
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
        with open(dest_path, 'wb') as out_file:
            out_file.write(response.read())


def main():
    print("[->] Integrating background images (Unsplash CDN) ...")
    success_count = 0

    for idx, (url, filename, split) in enumerate(BACKGROUND_SOURCES, 1):
        img_dest = IMAGES_DIR / split / filename
        lbl_dest = LABELS_DIR / split / (Path(filename).stem + ".txt")

        # Create directories if they don't exist
        img_dest.parent.mkdir(parents=True, exist_ok=True)
        lbl_dest.parent.mkdir(parents=True, exist_ok=True)

        # Force redownloading if file is tiny (corrupted)
        if img_dest.exists() and img_dest.stat().st_size > 1000 and lbl_dest.exists():
            print(f"  [{idx}/{len(BACKGROUND_SOURCES)}] {filename} already exists. Skipping.")
            success_count += 1
            continue

        print(f"  [{idx}/{len(BACKGROUND_SOURCES)}] Downloading {filename} to {split} ...")
        try:
            download_image(url, img_dest)
            
            # Write empty label file for background images
            lbl_dest.write_text("", encoding="utf-8")
            
            success_count += 1
            print(f"      [OK] Downloaded and wrote empty label.")
            
            # Sleep 0.2 seconds between downloads (Unsplash is high capacity, minor delay is safe)
            time.sleep(0.2)
        except Exception as e:
            print(f"      [ERROR] Failed to download {filename}: {e}")

    print("\n" + "=" * 55)
    print("  BACKGROUND IMAGE INTEGRATION COMPLETE")
    print("=" * 55)
    print(f"  Successfully added {success_count}/{len(BACKGROUND_SOURCES)} background images.")
    print(f"  Empty label files created to penalize false positive detections.")
    print("=" * 55)
    print("  Run scripts/train.py --resume or start a new training run to apply changes.")


if __name__ == "__main__":
    main()
