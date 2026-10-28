"""
ParticleVision V4
=================

F
Usage:
    python quick_label.py \\
        --output_dir /path/to/output \\
        --input_dir /path/to/SEM \\
        --examples_dir /path/to/Defects
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

import config
from build_prototypes import resolve_class_folder_name
from utils import load_rgb, logger


THUMBNAIL_SIZE = 205
GRID_COLUMNS = 5
GRID_ROWS = 3
CONTEXT_PAD_RATIO = 0.45
MIN_CONTEXT_PX = 120


# ============================================================
# CROP EXTRACTION
# ============================================================

def crop_with_context(image_rgb, bbox_x, bbox_y, bbox_w, bbox_h):
   

    h, w = image_rgb.shape[:2]

    pad_x = max(int(bbox_w * CONTEXT_PAD_RATIO), (MIN_CONTEXT_PX - bbox_w) // 2, 8)
    pad_y = max(int(bbox_h * CONTEXT_PAD_RATIO), (MIN_CONTEXT_PX - bbox_h) // 2, 8)

    x0 = max(0, bbox_x - pad_x)
    y0 = max(0, bbox_y - pad_y)
    x1 = min(w, bbox_x + bbox_w + pad_x)
    y1 = min(h, bbox_y + bbox_h + pad_y)

    clean = image_rgb[y0:y1, x0:x1].copy()

    marked = clean.copy()
    cv2.rectangle(
        marked,
        (bbox_x - x0, bbox_y - y0),
        (bbox_x - x0 + bbox_w, bbox_y - y0 + bbox_h),
        (255, 210, 0), 1,
    )

    return clean, marked


def build_review_queue(output_dir: str, input_dir: str, order: str = "uncertain", limit: int = 240):

    features_csv = os.path.join(output_dir, "features.csv")

    if not os.path.exists(features_csv):
        raise SystemExit(f"No features.csv in {output_dir} — run the pipeline first.")

    df = pd.read_csv(features_csv)

    if "defect_label" not in df.columns or df["defect_label"].isna().all():
        raise SystemExit("features.csv has no predictions — train a classifier first.")

    df = df[df["defect_label"].isin(config.DEFECT_TAXONOMY)]

    if order == "uncertain":
        df = df.sort_values("confidence", ascending=True)
    else:
        df = df.sort_values("confidence", ascending=False)

    df = df.head(limit)

    items = []
    image_cache = {}

    for _, row in df.iterrows():

        image_name = str(row["image_name"])
        image_path = os.path.join(input_dir, image_name)

        if not os.path.exists(image_path):
            continue

        if image_name not in image_cache:
            image_cache[image_name] = load_rgb(image_path)

        clean, marked = crop_with_context(
            image_cache[image_name],
            int(row["bbox_x"]), int(row["bbox_y"]),
            int(row["bbox_width"]), int(row["bbox_height"]),
        )

        items.append({
            "crop": marked,          # shown in the review grid
            "clean_crop": clean,     # what actually gets saved as training data
            "predicted": row["defect_label"],
            "confidence": float(row.get("confidence", 0.0) or 0.0),
            "particle_id": int(row["particle_id"]),
            "image_name": image_name,
            "image_path": image_path,
            "bbox": (int(row["bbox_x"]), int(row["bbox_y"]),
                     int(row["bbox_width"]), int(row["bbox_height"])),
        })

    return items


# ============================================================
# GUI
# ============================================================

class QuickLabeler(tk.Tk):

    def __init__(self, items, examples_dir):
        super().__init__()

        self.title("Quick Label — click only the WRONG ones")
        self.geometry("1400x960")

        self.examples_dir = examples_dir
        self.saved_count = 0
        self._photos = []

        # Group by predicted class so each page shows one claim.
        self.pages = []
        by_class = {}

        for item in items:
            by_class.setdefault(item["predicted"], []).append(item)

        page_size = GRID_COLUMNS * GRID_ROWS

        for class_name, class_items in by_class.items():
            for start in range(0, len(class_items), page_size):
                self.pages.append((class_name, class_items[start:start + page_size]))

        self.page_index = 0
        self.rejected = set()

        self.header = ttk.Label(self, text="", font=("", 15, "bold"))
        self.header.pack(anchor="w", padx=14, pady=(12, 2))

        self.subheader = ttk.Label(self, foreground="#6b6a66", wraplength=1200, justify="left")
        self.subheader.pack(anchor="w", padx=14, pady=(0, 8))

        self.grid_frame = ttk.Frame(self, padding=8)
        self.grid_frame.pack(fill="both", expand=True)

        controls = ttk.Frame(self, padding=10)
        controls.pack(fill="x")

        ttk.Button(controls, text="Confirm rest & continue  →", command=self.commit_page).pack(side="right")
        ttk.Button(controls, text="Skip this page", command=self.skip_page).pack(side="right", padx=(0, 8))

        self.status = ttk.Label(controls, text="", foreground="#6b6a66")
        self.status.pack(side="left")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.show_page()

    def show_page(self):

        for widget in self.grid_frame.winfo_children():
            widget.destroy()

        self._photos = []
        self.rejected = set()

        if self.page_index >= len(self.pages):
            self.finish()
            return

        class_name, items = self.pages[self.page_index]

        self.header.configure(text=f"All of these were predicted:   {class_name.replace('_', ' ')}")
        self.subheader.configure(
            text="LEFT-CLICK any that are NOT this class (they turn red), then "
                 "\"Confirm rest & continue\" — everything unmarked is saved as this class.   "
                 "RIGHT-CLICK (or double-click) any particle to see it large, with the full SEM "
                 "image for context.   The yellow box marks the particle being judged."
        )
        self.status.configure(
            text=f"Page {self.page_index + 1} of {len(self.pages)}  ·  {self.saved_count} saved so far"
        )

        for i, item in enumerate(items):

            image = Image.fromarray(item["crop"])
            image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self._photos.append(photo)

            cell = tk.Frame(self.grid_frame, bd=3, relief="flat", bg="#fcfcfb")
            cell.grid(row=i // GRID_COLUMNS, column=i % GRID_COLUMNS, padx=5, pady=5)

            label = tk.Label(cell, image=photo, bd=0)
            label.pack()

            confidence_label = tk.Label(
                cell, text=f"{item['confidence']:.0%}", font=("", 8),
                fg="#6b6a66", bg="#fcfcfb",
            )
            confidence_label.pack()

            for widget in (cell, label, confidence_label):
                widget.bind("<Button-1>", lambda _e, idx=i, c=cell: self.toggle(idx, c))
            
                widget.bind("<Button-2>", lambda _e, it=item: self.show_detail(it))
                widget.bind("<Button-3>", lambda _e, it=item: self.show_detail(it))
                widget.bind("<Double-Button-1>", lambda _e, it=item: self.show_detail(it))

        for column in range(GRID_COLUMNS):
            self.grid_frame.columnconfigure(column, weight=1)

    def toggle(self, index, cell):

        if index in self.rejected:
            self.rejected.discard(index)
            cell.configure(bg="#fcfcfb", relief="flat")
        else:
            self.rejected.add(index)
            cell.configure(bg="#e34948", relief="solid")

    def show_detail(self, item):
        
        window = tk.Toplevel(self)
        window.title(
            f"{item['image_name']}  ·  particle {item['particle_id']}  ·  "
            f"predicted {item['predicted'].replace('_', ' ')} ({item['confidence']:.0%})"
        )
        window.geometry("1500x820")

        holder = ttk.Frame(window, padding=10)
        holder.pack(fill="both", expand=True)

        left = ttk.Frame(holder)
        left.pack(side="left", padx=(0, 12))

        ttk.Label(left, text="This particle (zoomed)", font=("", 11, "bold")).pack(anchor="w")

        big = Image.fromarray(item["crop"])
        big = big.resize(
            (min(680, big.width * 4), min(680, big.height * 4)), Image.LANCZOS,
        )
        big_photo = ImageTk.PhotoImage(big)
        tk.Label(left, image=big_photo, bd=0).pack(pady=(6, 0))

        right = ttk.Frame(holder)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(right, text="Where it sits in the full image", font=("", 11, "bold")).pack(anchor="w")

        full = load_rgb(item["image_path"]).copy()
        x, y, w, h = item["bbox"]
        pad = 6
        cv2.rectangle(
            full, (max(0, x - pad), max(0, y - pad)), (x + w + pad, y + h + pad),
            (255, 210, 0), 3,
        )

        full_image = Image.fromarray(full)
        full_image.thumbnail((740, 700), Image.LANCZOS)
        full_photo = ImageTk.PhotoImage(full_image)
        tk.Label(right, image=full_photo, bd=0).pack(anchor="w", pady=(6, 0))

        window._photos = (big_photo, full_photo)

        ttk.Button(window, text="Close", command=window.destroy).pack(pady=(0, 10))
        window.bind("<Escape>", lambda _e: window.destroy())

    def commit_page(self):

      
        if self.page_index >= len(self.pages):
            self.finish()
            return

        class_name, items = self.pages[self.page_index]

        confirmed = [item for i, item in enumerate(items) if i not in self.rejected]
        wrong = [item for i, item in enumerate(items) if i in self.rejected]

        for item in confirmed:
            self._save_crop(item, class_name)

        self.page_index += 1

        if wrong:
            self._reassign(wrong)
        else:
            self.show_page()

    def skip_page(self):

        if self.page_index >= len(self.pages):
            self.finish()
            return

        self.page_index += 1
        self.show_page()

    def _reassign(self, wrong_items):
        """Second pass: assign the real class to the ones marked wrong."""

        window = tk.Toplevel(self)
        window.title("What are these actually?")
        window.geometry("900x900")

        window.transient(self)
        window.grab_set()

        state = {"position": 0, "photo": None}

        header = ttk.Label(window, text="", font=("", 13, "bold"))
        header.pack(anchor="w", padx=12, pady=(12, 6))

        image_label = ttk.Label(window)
        image_label.pack(pady=(0, 8))

        button_frame = ttk.Frame(window, padding=8)
        button_frame.pack(fill="x")

        def show():

            if state["position"] >= len(wrong_items):
                window.destroy()
                self.show_page()
                return

            item = wrong_items[state["position"]]

            image = Image.fromarray(item["crop"])
            image = image.resize((min(600, image.width*4), min(600, image.height*4)), Image.LANCZOS)
            state["photo"] = ImageTk.PhotoImage(image)
            image_label.configure(image=state["photo"])

            header.configure(
                text=f"{state['position'] + 1} of {len(wrong_items)}  —  "
                     f"predicted {item['predicted'].replace('_', ' ')}, what is it really?"
            )

        def assign(class_name):

            item = wrong_items[state["position"]]

            if class_name is not None:
                self._save_crop(item, class_name)

            state["position"] += 1
            show()

        for i, class_name in enumerate(config.DEFECT_TAXONOMY):
            ttk.Button(
                button_frame, text=class_name.replace("_", " "),
                command=lambda c=class_name: assign(c),
            ).grid(row=i // 2, column=i % 2, sticky="ew", padx=4, pady=3)

        for column in range(2):
            button_frame.columnconfigure(column, weight=1)

        ttk.Button(window, text="Discard (bad segmentation / unsure)", command=lambda: assign(None)).pack(pady=(6, 12))

        show()

    def _class_folder(self, class_name):
   
        for entry in sorted(os.listdir(self.examples_dir)):

            path = os.path.join(self.examples_dir, entry)

            if os.path.isdir(path) and resolve_class_folder_name(entry) == class_name:
                return path

       
        path = os.path.join(self.examples_dir, class_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _save_crop(self, item, class_name):
        """Writes the crop into Defects/<Class>/ so build_prototypes.py picks it up."""

        folder = self._class_folder(class_name)

        base = os.path.splitext(item["image_name"])[0]
        filename = f"auto_{base}_p{item['particle_id']}.png"
        path = os.path.join(folder, filename)

        cv2.imwrite(path, cv2.cvtColor(item["clean_crop"], cv2.COLOR_RGB2BGR))

        self.saved_count += 1

    def finish(self):

        messagebox.showinfo(
            "Quick Label",
            f"Done — {self.saved_count} new labeled example(s) saved into:\n{self.examples_dir}\n\n"
            "Next: rebuild prototypes and retrain to use them.",
        )
        self.destroy()

    def _on_close(self):
        print(f"[INFO] {self.saved_count} new example(s) saved to {self.examples_dir}")
        self.destroy()


def main():

    parser = argparse.ArgumentParser(description="Fast bulk labeling — verify predictions instead of choosing classes.")
    parser.add_argument("--output_dir", required=True, help="Pipeline output folder (needs features.csv).")
    parser.add_argument("--input_dir", required=True, help="Folder of the source SEM images.")
    parser.add_argument("--examples_dir", required=True, help="Your Defects/ folder — new crops are saved here.")
    parser.add_argument("--order", default="uncertain", choices=["uncertain", "confident"],
                        help="uncertain (default) = most informative first; confident = fastest bulk confirmation.")
    parser.add_argument("--limit", type=int, default=240, help="Max particles to queue up.")

    args = parser.parse_args()

    items = build_review_queue(args.output_dir, args.input_dir, order=args.order, limit=args.limit)

    if not items:
        raise SystemExit("Nothing to review — no classified particles matched the current taxonomy.")

    logger.info(f"Queued {len(items)} particle(s) for review, ordered by {args.order}-first.")

    app = QuickLabeler(items, args.examples_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
