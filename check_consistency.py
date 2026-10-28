"""
ParticleVision V4
=================

Intra-Annotator Consistency Check

Measures how consistently YOU label the same particles — the single
most useful number to have before investing weeks in more labeling,
because it is the practical ceiling on classifier accuracy. If you
only agree with yourself 60% of the time, no model trained on those
labels can honestly exceed ~60%: you'd be asking it to reproduce a
signal that isn't consistent in the data.

How it works: takes crops you have ALREADY sorted into class folders
(Defects/<Class>/), shuffles a random sample, and shows them one at a
time with the original label hidden. You re-label each one. At the
end it reports raw agreement plus Cohen's Kappa (chance-corrected —
the citable one), and saves a per-class breakdown so you can see
exactly WHICH classes you're inconsistent about.

That per-class breakdown is directly actionable: two classes you
routinely confuse with each other are usually two classes that should
be merged (as Molten/Bonded Satellite already were), not two classes
the model is failing at.

Usage:
    python check_consistency.py --examples_dir /path/to/Defects --n 40

Keyboard: press 1-8 to pick a class, or click the buttons.
"""

from __future__ import annotations

import argparse
import os
import random

import tkinter as tk
from tkinter import ttk

import pandas as pd
from PIL import Image, ImageTk
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import config
from build_prototypes import resolve_class_folder_name
from utils import list_images


REPORT_FILENAME = "consistency_report.csv"
DISPLAY_SIZE = (520, 520)


def collect_labeled_crops(examples_dir: str):
    """-> list of (image_path, original_class_label)"""

    items = []

    for folder_name in sorted(os.listdir(examples_dir)):

        folder = os.path.join(examples_dir, folder_name)

        if not os.path.isdir(folder):
            continue

        resolved = resolve_class_folder_name(folder_name)

        if resolved is None:
            continue

        for path in list_images(folder):
            items.append((str(path), resolved))

    return items


class ConsistencyChecker(tk.Tk):

    def __init__(self, samples, output_dir):
        super().__init__()

        self.title("Consistency Check — re-label these, original labels hidden")
        self.geometry("900x820")

        self.samples = samples
        self.output_dir = output_dir
        self.position = 0
        self.responses = []
        self._photo = None

        self.progress_label = ttk.Label(self, text="", font=("", 12, "bold"))
        self.progress_label.pack(anchor="w", padx=14, pady=(12, 2))

        ttk.Label(
            self,
            text="Label each particle as you normally would. Your original label is hidden — "
                 "answer from what you see, not what you remember.",
            foreground="#6b6a66", wraplength=860, justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 10))

        self.image_label = ttk.Label(self)
        self.image_label.pack(pady=(0, 10))

        button_frame = ttk.Frame(self, padding=10)
        button_frame.pack(fill="x")

        for i, class_name in enumerate(config.DEFECT_TAXONOMY):

            button = ttk.Button(
                button_frame,
                text=f"{i + 1}.  {class_name.replace('_', ' ')}",
                command=lambda c=class_name: self.record(c),
            )
            button.grid(row=i // 2, column=i % 2, sticky="ew", padx=4, pady=3)

            self.bind(str(i + 1), lambda _e, c=class_name: self.record(c))

        for column in range(2):
            button_frame.columnconfigure(column, weight=1)

        ttk.Button(self, text="Skip (unsure)", command=lambda: self.record(None)).pack(pady=(4, 12))

        self.show_current()

    def show_current(self):

        if self.position >= len(self.samples):
            self.finish()
            return

        path, _original = self.samples[self.position]

        image = Image.open(path).convert("RGB")
        image.thumbnail(DISPLAY_SIZE, Image.LANCZOS)

        self._photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self._photo)

        self.progress_label.configure(
            text=f"Particle {self.position + 1} of {len(self.samples)}"
        )

    def record(self, class_name):

        if self.position >= len(self.samples):
            return

        path, original = self.samples[self.position]

        self.responses.append({
            "source_path": path,
            "original_label": original,
            "relabel": class_name,
        })

        self.position += 1
        self.show_current()

    def finish(self):

        df = pd.DataFrame(self.responses)
        df.to_csv(os.path.join(self.output_dir, REPORT_FILENAME), index=False)

        answered = df[df["relabel"].notna()]

        print()
        print("=" * 60)
        print("INTRA-ANNOTATOR CONSISTENCY")
        print("=" * 60)

        if len(answered) == 0:
            print("No particles were labeled — nothing to report.")
            self.destroy()
            return

        agreement = (answered["original_label"] == answered["relabel"]).mean()
        kappa = cohen_kappa_score(answered["original_label"], answered["relabel"])

        print(f"Particles re-labeled : {len(answered)} (skipped {len(df) - len(answered)})")
        print(f"Raw agreement        : {agreement:.1%}")
        print(f"Cohen's Kappa        : {kappa:.3f}  ({_kappa_verdict(kappa)})")
        print()
        print("This is roughly the CEILING on classifier accuracy for this taxonomy.")
        print()

        print("Where you disagreed with yourself:")

        disagreements = answered[answered["original_label"] != answered["relabel"]]

        if len(disagreements) == 0:
            print("  (none — fully consistent)")
        else:
            pair_counts = (
                disagreements.groupby(["original_label", "relabel"]).size()
                .sort_values(ascending=False)
            )
            for (original, relabel), count in pair_counts.items():
                print(f"  {original} -> {relabel}: {count}x")
            print()
            print("Any pair you confuse repeatedly is a candidate for MERGING —")
            print("if you can't separate them reliably, a model can't either.")

        print()
        print(f"Full per-particle record saved to: {os.path.join(self.output_dir, REPORT_FILENAME)}")

        self.destroy()


def _kappa_verdict(kappa: float) -> str:
    """Landis & Koch (1977) interpretation scale — the standard citable one."""

    if kappa < 0.00:
        return "poor"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def main():

    parser = argparse.ArgumentParser(description="Measure how consistently you label the same particles.")
    parser.add_argument("--examples_dir", required=True, help="Folder of class subfolders (your Defects/ folder).")
    parser.add_argument("--output_dir", default=".", help="Where consistency_report.csv is written.")
    parser.add_argument("--n", type=int, default=40, help="How many particles to re-label (default 40).")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle seed. Leave unset for a different sample each run.")

    args = parser.parse_args()

    items = collect_labeled_crops(args.examples_dir)

    if not items:
        raise SystemExit(f"No labeled crops found under {args.examples_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(items)

    samples = items[: min(args.n, len(items))]

    print(f"[INFO] Re-labeling {len(samples)} of {len(items)} available crops. Original labels are hidden.")

    os.makedirs(args.output_dir, exist_ok=True)

    app = ConsistencyChecker(samples, args.output_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
