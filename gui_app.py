"""
ParticleVision V4
=================

GUI — 5 tabs: Run Pipeline, Defect Count, Compare Models, Results,
Image Preview. Launched the same way as before: `python gui_app.py`
(or your existing debugpy run configuration) — same `main()` /
`if __name__ == "__main__"` convention.

No per-particle labeling tab: prototype labeling happens by hand-
sorting crops into 10 folders (see build_prototypes.py), pointed to
from the Run Pipeline tab's "Example folders" field, not inside this
app.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import gc
import platform
import queue
import sys
import threading
import traceback

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

import matplotlib
matplotlib.use("Agg")

from PIL import Image, ImageTk
import joblib
import pandas as pd

import config
import main as pipeline
from utils import list_images, load_rgb


APP_TITLE = "ParticleVision V4"

RESULTS_PANEL_SIZE = (6.6, 5.2)

SURFACE_COLOR = (252, 252, 251)

REAL_PATH_DEFAULTS = {
    "input_dir": "SEM",
    "output_dir": "output",
    "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
    "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


# ============================================================
# SHARED WIDGET HELPERS
# ============================================================

def make_scrollable(parent, width=380, padding=10):


    outer = ttk.Frame(parent)
    outer.pack(side="left", fill="both", expand=True)

    canvas = tk.Canvas(outer, width=width, highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas, padding=padding)

    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_canvas_configure(event):
        canvas.itemconfig(window_id, width=event.width)

    canvas.bind("<Configure>", _on_canvas_configure)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    is_mac = platform.system() == "Darwin"

    def _on_mousewheel(event):

        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
        elif is_mac:
            canvas.yview_scroll(int(-1 * event.delta), "units")
        else:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_wheel(_event):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

    def _unbind_wheel(_event):
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas.bind("<Enter>", _bind_wheel)
    canvas.bind("<Leave>", _unbind_wheel)

    return inner


def _draw_tk_rounded_swatch(canvas, x0, y0, x1, y1, color, radius=4):


    r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)

    points = [
        x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
        x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
        x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
    ]

    canvas.create_polygon(points, smooth=True, fill=color, outline=color)


# ============================================================
# LOG REDIRECTION (pipeline runs in a background thread)
# ============================================================

class QueueWriter:
   

    def __init__(self, q: queue.Queue):
        self.queue = q

    def write(self, message):
        if message.strip():
            self.queue.put(message)

    def flush(self):
        pass


# ============================================================
# MAIN WINDOW
# ============================================================

class ParticleVisionGUI(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1400x900")
        self.minsize(1100, 700)

        self.output_dir_var = tk.StringVar(value=self._default_path("output_dir"))
        self.last_output_dir = self.output_dir_var.get() or None

        self.log_queue = queue.Queue()
        self._pipeline_thread = None
        self._preview_photo = None  # kept alive to avoid premature GC while displayed

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.run_tab = ttk.Frame(self.notebook)
        self.count_tab = ttk.Frame(self.notebook)
        self.models_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.preview_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.run_tab, text="1. Run Pipeline")
        self.notebook.add(self.count_tab, text="2. Defect Count")
        self.notebook.add(self.models_tab, text="3. Compare Models")
        self.notebook.add(self.results_tab, text="4. Results")
        self.notebook.add(self.preview_tab, text="5. Image Preview")

        self._build_run_tab()
        self._build_count_tab()
        self._build_models_tab()
        self._build_results_tab()
        self._build_preview_tab()

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(150, self._poll_log_queue)

    def _default_path(self, key):
        path = REAL_PATH_DEFAULTS.get(key, "")
        return path if os.path.exists(path) else ""

    # ========================================================
    # TAB SWITCH -> AUTO REFRESH
    # ========================================================

    def _on_tab_changed(self, _event=None):

        current = self.notebook.select()

        if current == str(self.count_tab):
            self._refresh_count_tab()
        elif current == str(self.models_tab):
            self._refresh_models_tab()
        elif current == str(self.results_tab):
            self._refresh_results_tab()
        elif current == str(self.preview_tab):
            self._populate_preview_list()

    # ========================================================
    # TAB 1 — RUN PIPELINE
    # ========================================================

    def _build_run_tab(self):

        # Two columns, side by side, both fully visible without
        # scrolling: settings (fixed width) on the left, log on the
        # right. No card/border styling here, flat, plain frames.
        root = ttk.Frame(self.run_tab, padding=14)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root, width=440)
        left.pack(side="left", fill="y", padx=(0, 14))
        left.pack_propagate(False)

        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Run Pipeline", font=("", 15, "bold")).pack(anchor="w")
        ttk.Label(
            left,
            text="One click: segment (SAM2) -> physics-constrained features -> DINOv2 -> classify -> reusability.",
            foreground="#6b6a66", wraplength=420, justify="left",
        ).pack(anchor="w", pady=(2, 12))

        self.input_dir_var = tk.StringVar(value=self._default_path("input_dir"))
        self.output_dir_var = tk.StringVar(value=self._default_path("output_dir"))
        self.checkpoint_var = tk.StringVar(value=self._default_path("checkpoint"))
        self.model_cfg_var = tk.StringVar(value=self._default_path("model_cfg"))
        self.calibration_csv_var = tk.StringVar()
        self.fov_width_var = tk.StringVar()
 
        self.scale_bar_var = tk.StringVar(value="10.0")
        self.examples_dir_var = tk.StringVar()
        self.prototype_mpp_var = tk.StringVar()
        self.device_var = tk.StringVar(value="cpu")
        self.retrain_var = tk.BooleanVar(value=False)

        self._file_picker_row(left, "Input image folder", self.input_dir_var, kind="dir")
        self._file_picker_row(left, "Output folder", self.output_dir_var, kind="dir")
        self._file_picker_row(
            left, "SAM2 checkpoint (.pt)", self.checkpoint_var, kind="file",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        self._file_picker_row(
            left, "SAM2 model config (.yaml)", self.model_cfg_var, kind="file",
            filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
        )
        self._file_picker_row(
            left, "Calibration CSV (optional)", self.calibration_csv_var, kind="file",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )

        ttk.Label(
            left,
            text="Simpler than a CSV — read straight off the microscope's info bar (e.g. \"FOV:128x96µm\" -> 128).",
            font=("", 9), foreground="#888", wraplength=420, justify="left",
        ).pack(anchor="w", pady=(6, 0))
        self._labeled_entry_row(left, "Field of view width (µm)", self.fov_width_var)

        ttk.Label(
            left,
            text="Used if FOV width is blank — the labeled length (µm) of the scale bar shown in each "
                 "image, measured directly from its pixel length (not OCR, which was found unreliable).",
            font=("", 9), foreground="#888", wraplength=420, justify="left",
        ).pack(anchor="w", pady=(6, 0))
        self._labeled_entry_row(left, "Scale bar length (µm)", self.scale_bar_var)

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Label(left, text="Classifier (optional)", font=("", 11, "bold")).pack(anchor="w")
        ttk.Label(
            left,
            text="Point to your example folders once to train the classifier. "
                 "Leave blank to reuse whatever classifier already exists in the output folder.",
            foreground="#6b6a66", wraplength=420, justify="left",
        ).pack(anchor="w", pady=(2, 8))

        self._file_picker_row(left, "Example folders", self.examples_dir_var, kind="dir")
        self._labeled_entry_row(left, "Example crops calibration (µm/pixel)", self.prototype_mpp_var)
        ttk.Checkbutton(left, text="Retrain even if a classifier already exists", variable=self.retrain_var).pack(anchor="w", pady=(4, 0))

        ttk.Separator(left).pack(fill="x", pady=10)

        row = ttk.Frame(left)
        row.pack(fill="x", pady=(2, 2))
        ttk.Label(row, text="Device").pack(side="left")
        ttk.Combobox(row, textvariable=self.device_var, values=["cpu", "cuda", "mps"], width=8, state="readonly").pack(side="left", padx=(8, 0))

        self.run_button = ttk.Button(left, text="Run Pipeline", command=self.run_pipeline)
        self.run_button.pack(anchor="w", pady=(14, 0))

        ttk.Label(right, text="Log", font=("", 11, "bold")).pack(anchor="w")
        self.log_text = tk.Text(right, wrap="word", state="disabled")
        log_scrollbar = ttk.Scrollbar(right, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True, pady=(6, 0))
        log_scrollbar.pack(side="left", fill="y", pady=(6, 0))

    def _file_picker_row(self, parent, label, var, kind="dir", filetypes=None):

        container = ttk.Frame(parent)
        container.pack(fill="x", pady=(6, 2))
        ttk.Label(container, text=label, anchor="w").pack(fill="x")

        row = ttk.Frame(container)
        row.pack(fill="x", pady=(2, 0))

        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True)

        def browse():
            if kind == "dir":
                path = filedialog.askdirectory()
            else:
                path = filedialog.askopenfilename(filetypes=filetypes or [("All files", "*.*")])
            if path:
                var.set(path)

        ttk.Button(row, text="Browse...", command=browse).pack(side="left", padx=(6, 0))

    def _labeled_entry_row(self, parent, label, var):

        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(4, 2))
        ttk.Label(row, text=label, width=32, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=var, width=16).pack(side="left")

    def _log(self, message: str):

        self.log_text.configure(state="normal")
        self.log_text.insert("end", message if message.endswith("\n") else message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _poll_log_queue(self):

        try:
            while True:
                message = self.log_queue.get_nowait()
                self._log(message)
        except queue.Empty:
            pass

        self.after(150, self._poll_log_queue)

    def run_pipeline(self):

        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            messagebox.showinfo(APP_TITLE, "Pipeline is already running.")
            return

        if not self.input_dir_var.get() or not self.output_dir_var.get():
            messagebox.showerror(APP_TITLE, "Input and output folders are required.")
            return

        if not self.checkpoint_var.get() or not self.model_cfg_var.get():
            messagebox.showerror(APP_TITLE, "SAM2 checkpoint and model config are required.")
            return

        examples_dir = self.examples_dir_var.get().strip() or None

        if examples_dir and not self.prototype_mpp_var.get().strip():
            messagebox.showerror(APP_TITLE, "Example-folder calibration (µm/pixel) is required when example folders are set.")
            return

        args = argparse_namespace(
            input_dir=self.input_dir_var.get(),
            output_dir=self.output_dir_var.get(),
            checkpoint=self.checkpoint_var.get(),
            model_cfg=self.model_cfg_var.get(),
            device=self.device_var.get(),
            calibration_csv=self.calibration_csv_var.get().strip() or None,
            fov_width_um=_to_float(self.fov_width_var.get()),
            scale_bar_um=_to_float(self.scale_bar_var.get()),
            examples_dir=examples_dir,
            prototype_microns_per_pixel=_to_float(self.prototype_mpp_var.get()),
            retrain_classifier=self.retrain_var.get(),
        )

        self.run_button.configure(state="disabled")
        self.last_output_dir = args.output_dir

        def worker():

            old_stdout = sys.stdout
            sys.stdout = QueueWriter(self.log_queue)

            try:
                pipeline.run_pipeline(args)
                self.log_queue.put("[DONE] Pipeline finished successfully.")
            except Exception:
                self.log_queue.put("[ERROR] Pipeline failed:\n" + traceback.format_exc())
            finally:
                sys.stdout = old_stdout
                self.after(0, lambda: self.run_button.configure(state="normal"))
                self.after(0, self._on_tab_changed)

        self._pipeline_thread = threading.Thread(target=worker, daemon=True)
        self._pipeline_thread.start()

    # ========================================================
    # TAB 2 — DEFECT COUNT
    # ========================================================

    def _build_count_tab(self):

        inner = make_scrollable(self.count_tab, width=1300)
        self.count_tab_inner = inner

        ttk.Label(inner, text="Defect Count", font=("", 15, "bold")).pack(anchor="w")
        ttk.Label(inner, text="Per-class particle counts and reusability summary for the current output folder.", foreground="#6b6a66").pack(anchor="w", pady=(2, 12))

        self.count_status_label = ttk.Label(inner, text="Run the pipeline first, or switch to this tab after a run.")
        self.count_status_label.pack(anchor="w")

        self.count_tables_frame = ttk.Frame(inner)
        self.count_tables_frame.pack(fill="both", expand=True, pady=(10, 0))

    def _refresh_count_tab(self):

        for widget in self.count_tables_frame.winfo_children():
            widget.destroy()

        output_dir = self.output_dir_var.get() or self.last_output_dir

        if not output_dir:
            self.count_status_label.configure(text="No output folder set yet.")
            return

        features_csv = os.path.join(output_dir, "features.csv")

        if not os.path.exists(features_csv):
            self.count_status_label.configure(text=f"No features.csv in {output_dir} yet — run the pipeline first.")
            return

        df = pd.read_csv(features_csv)
        self.count_status_label.configure(text=f"{len(df)} particle(s) across {df['image_name'].nunique()} image(s).")

        if "defect_label" in df.columns and df["defect_label"].notna().any():

            counts = df["defect_label"].value_counts()
            order = [c for c in config.DEFECT_TAXONOMY if c in counts.index]
            count_df = pd.DataFrame({
                "Class": [c.replace("_", " ") for c in order],
                "Count": [int(counts[c]) for c in order],
                "Percent": [f"{100.0 * counts[c] / len(df):.1f}%" for c in order],
            })
            self._add_table(self.count_tables_frame, "Class distribution", count_df)

        else:
            ttk.Label(self.count_tables_frame, text="No classifier has been trained yet — particles are unclassified. Provide example folders on the Run Pipeline tab.", foreground="#a33").pack(anchor="w", pady=(4, 10))

        summary_csv = os.path.join(output_dir, "reusability_summary.csv")

        if os.path.exists(summary_csv):
            summary_df = pd.read_csv(summary_csv)
            display_cols = [c for c in summary_df.columns if not c.startswith("pct_")]
            self._add_table(self.count_tables_frame, "Reusability summary (per image)", summary_df[display_cols].round(2))

    def _add_table(self, parent, title, df: pd.DataFrame, height=10):

        section = ttk.Frame(parent)
        section.pack(fill="x", pady=(0, 16), anchor="n")

        ttk.Label(section, text=title, font=("", 11, "bold")).pack(anchor="w")

        columns = list(df.columns)
        tree = ttk.Treeview(section, columns=columns, show="headings", height=min(height, max(3, len(df))))

        for col in columns:
            tree.heading(col, text=col.replace("_", " ").title())
            tree.column(col, width=max(90, min(220, 14 * len(col))), anchor="center")

        for _, row in df.iterrows():
            tree.insert("", "end", values=[row[c] for c in columns])

        tree.pack(fill="x", pady=(4, 0))

    # ========================================================
    # TAB 3 — COMPARE MODELS (tables only)
    # ========================================================

    def _build_models_tab(self):

        inner = make_scrollable(self.models_tab, width=1300)
        self.models_tab_inner = inner

        ttk.Label(inner, text="Compare Models", font=("", 15, "bold")).pack(anchor="w")
        ttk.Label(inner, text="Cross-validated on your labeled prototype set. Tables only — see the Results tab for charts.", foreground="#6b6a66").pack(anchor="w", pady=(2, 12))

        self.models_status_label = ttk.Label(inner, text="Run the pipeline (with example folders) first.")
        self.models_status_label.pack(anchor="w")

        self.models_tables_frame = ttk.Frame(inner)
        self.models_tables_frame.pack(fill="both", expand=True, pady=(10, 0))

    def _refresh_models_tab(self):

        for widget in self.models_tables_frame.winfo_children():
            widget.destroy()

        output_dir = self.output_dir_var.get() or self.last_output_dir
        classifier_dir = os.path.join(output_dir, "classifier") if output_dir else None

        if not classifier_dir or not os.path.exists(classifier_dir):
            self.models_status_label.configure(text="No trained classifier yet.")
            return

        best_model_path = os.path.join(classifier_dir, "best_model_name.joblib")
        best_model_name = joblib.load(best_model_path) if os.path.exists(best_model_path) else None

        self.models_status_label.configure(
            text=f"Classifier: {classifier_dir}"
            + (f"  —  best model (cross-validated): {best_model_name}" if best_model_name else "")
        )

        comparison_path = os.path.join(classifier_dir, "model_comparison.csv")

        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path).round(3)
            self._add_table(self.models_tables_frame, "Model comparison (cross-validated, all models)", comparison_df, height=12)

        per_class_path = os.path.join(classifier_dir, "per_class_metrics.csv")

        if os.path.exists(per_class_path):
            per_class_df = pd.read_csv(per_class_path)
            if best_model_name and "model" in per_class_df.columns:
                per_class_df = per_class_df[per_class_df["model"] == best_model_name].drop(columns=["model"])
            numeric_cols = per_class_df.select_dtypes(include="number").columns
            per_class_df[numeric_cols] = per_class_df[numeric_cols].round(3)
            self._add_table(self.models_tables_frame, f"Per-class precision / recall / F1 ({best_model_name or 'best model'})", per_class_df, height=12)

        confusion_path = os.path.join(classifier_dir, "confusion_matrix.csv")

        if os.path.exists(confusion_path):
            confusion_df = pd.read_csv(confusion_path, index_col=0).reset_index().rename(columns={"index": "true_class"})
            self._add_table(self.models_tables_frame, f"Confusion matrix ({best_model_name or 'best model'})", confusion_df, height=12)

        ablation_path = os.path.join(classifier_dir, "ablation.csv")

        if os.path.exists(ablation_path):
            ablation_df = pd.read_csv(ablation_path).round(3)
            self._add_table(self.models_tables_frame, "Feature ablation", ablation_df, height=12)

        importance_path = os.path.join(classifier_dir, "feature_importance.csv")

        if os.path.exists(importance_path):
            importance_df = pd.read_csv(importance_path)
            if best_model_name and "model" in importance_df.columns and best_model_name in set(importance_df["model"]):
                importance_df = importance_df[importance_df["model"] == best_model_name].drop(columns=["model"])
            importance_df = importance_df.head(20).round(3)
            self._add_table(self.models_tables_frame, f"Top feature importances ({best_model_name or 'best model'})", importance_df, height=12)

    # ========================================================
    # TAB 4 — RESULTS (charts)
    # ========================================================

    def _build_results_tab(self):

        inner = make_scrollable(self.results_tab, width=1360)
        self.results_tab_inner = inner

        ttk.Label(inner, text="Results", font=("", 15, "bold")).pack(anchor="w")
        ttk.Label(inner, text="Pipeline results are saved as CSV files in the output folder.", foreground="#6b6a66").pack(anchor="w", pady=(2, 12))

        self.results_status_label = ttk.Label(inner, text="Run the pipeline first.")
        self.results_status_label.pack(anchor="w")

        self.results_grid = ttk.Frame(inner)
        self.results_grid.pack(fill="both", expand=True, pady=(10, 0))

        self._results_photos = []  # kept alive while displayed

    def _refresh_results_tab(self):

        for widget in self.results_grid.winfo_children():
            widget.destroy()

        self._results_photos = []

        output_dir = self.output_dir_var.get() or self.last_output_dir

        if not output_dir:
            self.results_status_label.configure(text="No output folder set yet.")
            return

        figs = {}  # chart generation is not part of this release

        if not figs:
            self.results_status_label.configure(text=f"Results are written as CSV files to {output_dir}.")
            return

        self.results_status_label.configure(text=f"{len(figs)} chart(s).")

        fig_dir = os.path.join(output_dir, "figures")
        names = list(figs.keys())

        # 2 columns, each chart letterboxed into an IDENTICAL fixed
        # display box (never cropped, uniform size regardless of a
        # chart's own native aspect ratio) — box size is computed from
        # the canvas's ACTUAL current width, not a hardcoded guess, so
        # it always fits without needing horizontal scroll (which this
        # app doesn't wire up) regardless of window size.
        columns = 2
        cell_padding = 8

        self.update_idletasks()
        available_width = max(self.results_tab_inner.winfo_width(), 700)

        box_w = max(360, available_width // columns - cell_padding * 2)
        box_h = int(box_w / (RESULTS_PANEL_SIZE[0] / RESULTS_PANEL_SIZE[1]))

        for i, name in enumerate(names):

            png_path = os.path.join(fig_dir, f"{name}.png")

            if not os.path.exists(png_path):
                continue

            image = Image.open(png_path).convert("RGB")
            image.thumbnail((box_w, box_h), Image.LANCZOS)

            canvas_img = Image.new("RGB", (box_w, box_h), SURFACE_COLOR)
            paste_x = (box_w - image.width) // 2
            paste_y = (box_h - image.height) // 2
            canvas_img.paste(image, (paste_x, paste_y))

            photo = ImageTk.PhotoImage(canvas_img)
            self._results_photos.append(photo)

            panel_letter = chr(ord("A") + i) if i < 26 else f"A{i - 25}"

            cell = ttk.Frame(self.results_grid, padding=cell_padding)
            cell.grid(row=i // columns, column=i % columns, sticky="nsew")

            ttk.Label(cell, text=f"{panel_letter}.  {name.replace('_', ' ').title()}", font=("", 10, "bold")).pack(anchor="w")
            ttk.Label(cell, image=photo).pack(anchor="w", pady=(4, 0))

        for c in range(columns):
            self.results_grid.columnconfigure(c, weight=1)

    # ========================================================
    # TAB 5 — IMAGE PREVIEW
    # ========================================================

    def _build_preview_tab(self):

        left = ttk.Frame(self.preview_tab, padding=12, width=340)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        ttk.Label(left, text="Image Preview", font=("", 13, "bold")).pack(anchor="w")
        ttk.Label(left, text="Classification overlay for a segmented image.", foreground="#6b6a66", wraplength=310, justify="left").pack(anchor="w", pady=(2, 10))

        self.preview_listbox = tk.Listbox(left)
        self.preview_listbox.pack(fill="both", expand=True)
        self.preview_listbox.bind("<<ListboxSelect>>", lambda e: self.show_preview())

        ttk.Button(left, text="Refresh list", command=self._populate_preview_list).pack(fill="x", pady=(8, 8))

       
        self._build_legend_panel(left)

        right = ttk.Frame(self.preview_tab, padding=12)
        right.pack(side="left", fill="both", expand=True)

        self.preview_canvas_frame = ttk.Label(right)
        self.preview_canvas_frame.pack(fill="both", expand=True)

    def _build_legend_panel(self, parent):

        legend_frame = ttk.Frame(parent)
        legend_frame.pack(side="bottom", fill="x", pady=(8, 0))

        ttk.Separator(legend_frame).pack(fill="x", pady=(0, 8))
        ttk.Label(legend_frame, text="Legend", font=("", 10, "bold")).pack(anchor="w")

        columns = 3
        swatch = 16
        gap = 6
        cell_h = 28
    
        labels = list(config.DEFECT_TAXONOMY) + [config.BORDER_TRUNCATED_LABEL]
        display_names = {config.BORDER_TRUNCATED_LABEL: "Edge-truncated (excluded)"}
        rows = (len(labels) + columns - 1) // columns

        legend_font = tkfont.Font(font=("", 10))

        max_text_w = max(
            legend_font.measure(display_names.get(l, l.replace("_", " ")))
            for l in labels
        )
        cell_w = swatch + gap + max_text_w + gap

        canvas_w = cell_w * columns + 4

    
        canvas = tk.Canvas(
            legend_frame, width=canvas_w, height=rows * cell_h + 6,
            highlightthickness=1, highlightbackground="#d8d7d2", bg="#fcfcfb",
        )
        canvas.pack(anchor="w", pady=(4, 0))

        for i, label in enumerate(labels):

            col = i % columns
            row = i // columns

            x = col * cell_w + 2
            y = row * cell_h + 4

            if label == config.BORDER_TRUNCATED_LABEL:
                color = config.BORDER_TRUNCATED_COLOR
            else:
                color = config.DEFECT_COLORS.get(label, (140, 140, 140))

            hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

            _draw_tk_rounded_swatch(canvas, x, y, x + swatch, y + swatch, hex_color, radius=4)

            canvas.create_text(
                x + swatch + gap, y + swatch / 2,
                text=display_names.get(label, label.replace("_", " ")),
                anchor="w", font=legend_font, fill="#0b0b0b",
            )

        left_width = canvas_w + 24
        parent.configure(width=max(340, left_width))

    def _populate_preview_list(self):

        output_dir = self.output_dir_var.get() or self.last_output_dir
        input_dir = self.input_dir_var.get()

        self.preview_listbox.delete(0, "end")

        source_dir = input_dir if input_dir and os.path.isdir(input_dir) else output_dir

        if not source_dir or not os.path.isdir(source_dir):
            return

        for path in list_images(source_dir):
            self.preview_listbox.insert("end", os.path.basename(str(path)))

        self._preview_source_dir = source_dir

    def show_preview(self):

        selection = self.preview_listbox.curselection()

        if not selection:
            return

        image_name = self.preview_listbox.get(selection[0])
        image_path = os.path.join(getattr(self, "_preview_source_dir", ""), image_name)

        if not os.path.exists(image_path):
            return

        output_dir = self.output_dir_var.get() or self.last_output_dir
        features_csv = os.path.join(output_dir, "features.csv") if output_dir else None

        image_rgb = load_rgb(image_path)

        import visualization

        if features_csv and os.path.exists(features_csv):

            df = pd.read_csv(features_csv)
            df = df[df["image_name"] == image_name]

            if len(df) and "defect_label" in df.columns:

                label_map = pipeline.load_particle_label_map(output_dir, image_name)
                particles = _rows_to_display_particles(df, image_rgb.shape, label_map=label_map)
                overlay = visualization.draw_classification_overlay(image_rgb, particles)

            else:
                overlay = image_rgb.copy()

        else:
            overlay = image_rgb.copy()


        pil_image = Image.fromarray(overlay)
        pil_image.thumbnail((1000, 800), Image.LANCZOS)

        self._preview_photo = ImageTk.PhotoImage(pil_image)
        self.preview_canvas_frame.configure(image=self._preview_photo)

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def _on_close(self):

        try:
            self._preview_photo = None
            self._results_photos = []

            for attr in ("preview_canvas_frame",):
                widget = getattr(self, attr, None)
                if widget is not None and hasattr(widget, "configure"):
                    try:
                        widget.configure(image="")
                        widget.image = None
                    except tk.TclError:
                        pass

            import matplotlib.pyplot as plt
            plt.close("all")

            gc.collect()

        except Exception:
            pass

        finally:
            self.destroy()


# ============================================================
# HELPERS
# ============================================================

def _to_float(value: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def argparse_namespace(**kwargs):

    import argparse
    return argparse.Namespace(**kwargs)


def _rows_to_display_particles(df: pd.DataFrame, image_shape, label_map=None):


    import cv2
    import numpy as np
    from data_models import Particle, BoundingBox, Centroid, PredictionResult

    particles = []

    for _, row in df.iterrows():

        x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_width"]), int(row["bbox_height"])
        particle_id = int(row["particle_id"])

        mask = None

        if label_map is not None:
            mask = label_map == particle_id

            if not mask.any():
                mask = None  

        if mask is None:
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, (x + w // 2, y + h // 2), (max(w // 2, 1), max(h // 2, 1)), 0, 0, 360, 1, -1)
            mask = mask.astype(bool)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea) if contours else None

        particle = Particle(
            particle_id=particle_id, image_name=row["image_name"],
            mask=mask, contour=contour,
            bbox=BoundingBox(x, y, w, h),
            centroid=Centroid(row.get("centroid_x", x + w / 2), row.get("centroid_y", y + h / 2)),
        )
        particle.prediction = PredictionResult(label=row.get("defect_label", "Unknown"), confidence=row.get("confidence", 0.0) or 0.0)

        particles.append(particle)

    return particles


def main():
    app = ParticleVisionGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
