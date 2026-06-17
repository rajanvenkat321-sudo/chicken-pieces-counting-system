"""
convert_labels.py
-----------------
Converts all YOLO label files in the dataset from multi-class (0-5)
to single-class (all -> 0: chicken_piece).

Run ONCE before training:
  python scripts/convert_labels.py
"""

import sys
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_ROOT  = PROJECT_ROOT / "datasets" / "labels"

converted = 0
skipped   = 0

for split in ("train", "val", "test"):
    split_dir = LABELS_ROOT / split
    if not split_dir.exists():
        print(f"[!] Skipping missing dir: {split_dir}")
        continue

    txt_files = list(split_dir.glob("*.txt"))
    print(f"\n[->] Processing {split}: {len(txt_files)} files")

    for txt_path in txt_files:
        lines = txt_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        changed = False

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                if parts[0] != "0":
                    parts[0] = "0"
                    changed = True
                new_lines.append(" ".join(parts))
            else:
                print(f"  [!] Skipping malformed line in {txt_path.name}: {line!r}")

        txt_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        if changed:
            converted += 1
            print(f"  [OK] Converted: {txt_path.name}")
        else:
            skipped += 1
            print(f"  [==] Already 0: {txt_path.name}")

print(f"\n{'='*50}")
print(f"  Done! {converted} file(s) converted, {skipped} already correct.")
print(f"{'='*50}")
