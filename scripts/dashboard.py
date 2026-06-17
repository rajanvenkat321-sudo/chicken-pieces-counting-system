"""
dashboard.py
------------
Live chicken piece detection dashboard using Tkinter.
Select any image and see real-time detection results with bounding boxes,
piece count, and confidence scores.

Usage:
    python scripts/dashboard.py
"""

import sys
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except ImportError:
    raise ImportError("tkinter is required (included with Python)")

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
except ImportError:
    raise ImportError("Run: pip install Pillow")

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Run: pip install ultralytics")

import numpy as np

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "chicken_detect" / "weights" / "best.pt"
DATA_YAML = PROJECT_ROOT / "datasets" / "data.yaml"

# ─────────────────────────────────────────────
#  Color palette (dark theme)
# ─────────────────────────────────────────────
BG_DARK    = "#1a1a2e"   # deep navy background
BG_CARD    = "#16213e"   # card / sidebar background
BG_PANEL   = "#0f3460"   # header panel
ACCENT     = "#e94560"   # vivid red accent
ACCENT2    = "#f5a623"   # warm amber (for count badge)
TEXT_LIGHT = "#eaeaea"   # primary text
TEXT_DIM   = "#8892b0"   # muted text
TEXT_GREEN = "#64ffda"   # success / confidence green
BTN_HOVER  = "#c73652"   # button hover
FONT_FAMILY = "Helvetica"


class ChickenDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chicken Piece Detector  |  Live Demo")
        self.geometry("1200x750")
        self.minsize(900, 600)
        self.configure(bg=BG_DARK)

        self._model = None
        self._model_path = DEFAULT_WEIGHTS
        self._current_image_path = None
        self._photo = None          # keep reference to avoid GC
        self._inference_running = False
        self._pending_rerun = False

        self._conf_var = tk.DoubleVar(value=0.25)
        self._iou_var  = tk.DoubleVar(value=0.60)

        self._build_ui()
        self._load_model_async()

    # ─────────────────── UI Construction ──────────────────────
    def _build_ui(self):
        # ── Header bar ──────────────────────────────────────
        header = tk.Frame(self, bg=BG_PANEL, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="🍗  Chicken Piece Detector",
                 font=(FONT_FAMILY, 18, "bold"),
                 bg=BG_PANEL, fg=TEXT_LIGHT).pack(side="left", padx=20, pady=12)

        self._status_lbl = tk.Label(header, text="Loading model ...",
                                    font=(FONT_FAMILY, 10),
                                    bg=BG_PANEL, fg=TEXT_DIM)
        self._status_lbl.pack(side="right", padx=20)

        # ── Main layout: sidebar + canvas ────────────────────
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True)

        # Sidebar
        sidebar = tk.Frame(main, bg=BG_CARD, width=280)
        sidebar.pack(fill="y", side="left")
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Canvas area
        canvas_frame = tk.Frame(main, bg=BG_DARK)
        canvas_frame.pack(fill="both", expand=True, side="left")
        self._build_canvas(canvas_frame)

    def _build_sidebar(self, parent):
        pad = dict(padx=20, pady=8)

        # ── Section: Image selection ──────────────────────────
        tk.Label(parent, text="IMAGE", font=(FONT_FAMILY, 9, "bold"),
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w", padx=20, pady=(20, 4))

        self._select_btn = tk.Button(
            parent, text="  Select Image",
            font=(FONT_FAMILY, 11, "bold"),
            bg=ACCENT, fg="white", activebackground=BTN_HOVER,
            activeforeground="white", relief="flat", cursor="hand2",
            command=self._on_select_image, pady=10)
        self._select_btn.pack(fill="x", padx=20, pady=4)

        self._img_name_lbl = tk.Label(parent, text="No image selected",
                                      font=(FONT_FAMILY, 8), wraplength=240,
                                      bg=BG_CARD, fg=TEXT_DIM)
        self._img_name_lbl.pack(anchor="w", padx=20, pady=2)

        _separator(parent)

        # ── Section: Settings ────────────────────────────────
        tk.Label(parent, text="SETTINGS", font=(FONT_FAMILY, 9, "bold"),
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w", **pad)

        _slider_row(parent, "Confidence", self._conf_var, 0.01, 0.99,
                    on_change=lambda *_: self._rerun_if_image())
        _slider_row(parent, "IoU Threshold", self._iou_var, 0.1, 0.9,
                    on_change=lambda *_: self._rerun_if_image())

        self._autotune_btn = tk.Button(
            parent, text="⚡ Auto-Tune Thresholds",
            font=(FONT_FAMILY, 9, "bold"),
            bg=BG_PANEL, fg=TEXT_LIGHT, activebackground=BG_DARK,
            activeforeground="white", relief="flat", cursor="hand2",
            command=self._on_autotune_thresholds, pady=6)
        self._autotune_btn.pack(fill="x", padx=20, pady=8)

        _separator(parent)

        # ── Section: Results ─────────────────────────────────
        tk.Label(parent, text="RESULTS", font=(FONT_FAMILY, 9, "bold"),
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w", **pad)

        # Count badge
        badge_frame = tk.Frame(parent, bg=BG_CARD)
        badge_frame.pack(fill="x", padx=20, pady=4)

        tk.Label(badge_frame, text="Chicken Pieces",
                 font=(FONT_FAMILY, 10), bg=BG_CARD, fg=TEXT_LIGHT).pack(side="left")

        self._count_badge = tk.Label(badge_frame, text="—",
                                     font=(FONT_FAMILY, 14, "bold"),
                                     bg=ACCENT2, fg=BG_DARK,
                                     width=4, relief="flat")
        self._count_badge.pack(side="right")

        # Inference time
        self._time_lbl = tk.Label(parent, text="Inference: —",
                                  font=(FONT_FAMILY, 9), bg=BG_CARD, fg=TEXT_DIM)
        self._time_lbl.pack(anchor="w", padx=20)

        _separator(parent)

        # Detection list header
        tk.Label(parent, text="DETECTIONS", font=(FONT_FAMILY, 9, "bold"),
                 bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w", **pad)

        list_frame = tk.Frame(parent, bg=BG_CARD)
        list_frame.pack(fill="both", expand=True, padx=20, pady=4)

        self._det_list = tk.Text(
            list_frame, bg=BG_DARK, fg=TEXT_GREEN,
            font=("Courier", 9), relief="flat",
            state="disabled", wrap="none",
            bd=0, highlightthickness=0)
        self._det_list.pack(fill="both", expand=True)

    def _build_canvas(self, parent):
        # Placeholder label shown before an image is loaded
        self._canvas_label = tk.Label(
            parent, bg=BG_DARK,
            text="Select an image to begin detection",
            font=(FONT_FAMILY, 14), fg=TEXT_DIM)
        self._canvas_label.pack(fill="both", expand=True)

    # ─────────────────── Model loading ────────────────────────
    def _load_model_async(self):
        def _load():
            try:
                self._model = YOLO(str(self._model_path))
                self.after(0, lambda: self._set_status(
                    f"Model ready  |  {self._model_path.name}", color=TEXT_GREEN))
            except Exception as e:
                self.after(0, lambda: self._set_status(
                    f"Model error: {e}", color=ACCENT))

        threading.Thread(target=_load, daemon=True).start()

    # ─────────────────── Image selection ──────────────────────
    def _on_select_image(self):
        path = filedialog.askopenfilename(
            title="Select a chicken image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.JPEG *.PNG")])
        if path:
            self._current_image_path = path
            self._img_name_lbl.config(text=Path(path).name)
            self._run_detection(path)

    def _rerun_if_image(self):
        if self._current_image_path:
            if self._inference_running:
                self._pending_rerun = True
            else:
                self._run_detection(self._current_image_path)

    # ─────────────────── Inference ────────────────────────────
    def _run_detection(self, img_path: str):
        if self._inference_running:
            return
        if self._model is None:
            self._set_status("Model not loaded yet — please wait.", color=ACCENT)
            return

        self._inference_running = True
        self._set_status("Running inference ...", color=TEXT_DIM)
        self._select_btn.config(state="disabled")

        def _infer():
            try:
                t0 = time.perf_counter()
                results = self._model.predict(
                    source=img_path,
                    conf=self._conf_var.get(),
                    iou=self._iou_var.get(),
                    verbose=False,
                    single_cls=True,
                )
                elapsed = (time.perf_counter() - t0) * 1000   # ms
                result = results[0]
                self.after(0, lambda: self._update_ui(result, elapsed))
            except Exception as e:
                self.after(0, lambda: self._set_status(f"Error: {e}", color=ACCENT))
            finally:
                self._inference_running = False
                self.after(0, lambda: self._select_btn.config(state="normal"))
                if self._pending_rerun:
                    self._pending_rerun = False
                    self.after(50, lambda: self._run_detection(self._current_image_path))

        threading.Thread(target=_infer, daemon=True).start()

    # ─────────────────── UI update ────────────────────────────
    def _update_ui(self, result, elapsed_ms: float):
        # ── Annotated image via result.plot() (BGR numpy) ────
        annotated_bgr = result.plot(line_width=2, font_size=10)
        pil_img = Image.fromarray(annotated_bgr[..., ::-1])   # BGR → RGB

        # Resize to fit canvas while preserving aspect ratio
        pil_img = self._fit_image(pil_img)
        self._photo = ImageTk.PhotoImage(pil_img)

        self._canvas_label.config(image=self._photo, text="",
                                  bg=BG_DARK)

        # ── Count & time ─────────────────────────────────────
        boxes = result.boxes
        n = len(boxes) if boxes is not None else 0
        self._count_badge.config(text=str(n))
        self._time_lbl.config(text=f"Inference: {elapsed_ms:.1f} ms")
        self._set_status(f"Done  |  {n} chicken piece(s) detected", color=TEXT_GREEN)

        # ── Detection list ───────────────────────────────────
        self._det_list.config(state="normal")
        self._det_list.delete("1.0", "end")
        if n == 0:
            self._det_list.insert("end", "  No detections above threshold\n")
        else:
            for i, box in enumerate(boxes, 1):
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                w = x2 - x1
                h = y2 - y1
                line = (f"  [{i:02d}]  conf={conf:.2f}  "
                        f"({x1},{y1})  {w}x{h}px\n")
                self._det_list.insert("end", line)
        self._det_list.config(state="disabled")

    def _fit_image(self, img: Image.Image) -> Image.Image:
        """Resize image to fill the canvas label, maintaining aspect ratio."""
        cw = self.winfo_width() - 290    # subtract sidebar width
        ch = self.winfo_height() - 70    # subtract header height
        cw = max(cw, 300)
        ch = max(ch, 300)

        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        new_size = (int(iw * scale), int(ih * scale))
        return img.resize(new_size, Image.LANCZOS)

    # ─────────────────── Helpers ──────────────────────────────
    def _set_status(self, msg: str, color: str = TEXT_DIM):
        self._status_lbl.config(text=msg, fg=color)

    def _on_autotune_thresholds(self):
        if self._model is None:
            self._set_status("Please wait for model to load first", color=ACCENT)
            return

        # If an image is loaded → per-image analysis; otherwise → val-set fallback
        if self._current_image_path:
            self._autotune_for_image()
        else:
            self._autotune_from_val()

    # ─────────── Per-image auto-tuning ────────────────────────
    def _autotune_for_image(self):
        """Analyze the current image to find optimal conf & IoU thresholds."""
        self._autotune_btn.config(state="disabled", text="⚡ Analyzing image...")
        self._set_status("Analyzing image for optimal thresholds ...", color=TEXT_DIM)

        def _analyze():
            try:
                img_path = self._current_image_path

                # ── Step 1: Run at very low conf, almost no NMS suppression ──
                # This captures every possible candidate detection.
                results = self._model.predict(
                    source=img_path,
                    conf=0.01,       # catch everything
                    iou=0.99,        # suppress almost nothing
                    max_det=300,     # allow many detections
                    verbose=False,
                    single_cls=True,
                )
                result = results[0]
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    # No detections at all — use permissive defaults
                    self.after(0, lambda: self._apply_tuned_thresholds(0.10, 0.60,
                        info="No detections found — using permissive defaults"))
                    return

                confs = sorted([float(c) for c in boxes.conf], reverse=True)
                xyxy  = boxes.xyxy.cpu().numpy()

                # ── Step 2: Find optimal confidence (largest-gap method) ─────
                #  Sort confidences high→low, find the biggest drop.
                #  The gap separates real chicken from background noise.
                best_conf = 0.25
                if len(confs) >= 2:
                    gaps = []
                    for i in range(len(confs) - 1):
                        gap_size  = confs[i] - confs[i + 1]
                        threshold = confs[i + 1]  # just above the lower value
                        gaps.append((gap_size, threshold, i))

                    # Pick the largest gap that is meaningful (> 0.05)
                    gaps.sort(key=lambda g: g[0], reverse=True)
                    for gap_size, threshold, idx in gaps:
                        if gap_size >= 0.05:
                            # Place the threshold just above the lower cluster
                            best_conf = threshold + 0.01
                            break
                    else:
                        # No clear gap → use 70 % of the median confidence
                        best_conf = confs[len(confs) // 2] * 0.70
                elif len(confs) == 1:
                    best_conf = confs[0] * 0.50

                best_conf = round(max(0.05, min(0.85, best_conf)), 2)

                # ── Step 3: Filter boxes above the chosen conf ───────────────
                high_conf_indices = [
                    i for i, c in enumerate(boxes.conf) if float(c) >= best_conf
                ]
                high_conf_boxes = xyxy[high_conf_indices]
                n_kept = len(high_conf_boxes)

                # ── Step 4: Find optimal IoU (pairwise overlap analysis) ─────
                #  Compute the maximum IoU between any two valid boxes.
                #  Set the threshold ABOVE that max so NMS never merges them.
                max_pairwise_iou = 0.0
                if n_kept >= 2:
                    for i in range(n_kept):
                        for j in range(i + 1, n_kept):
                            iou_val = _compute_iou(high_conf_boxes[i],
                                                   high_conf_boxes[j])
                            if iou_val > max_pairwise_iou:
                                max_pairwise_iou = iou_val

                if max_pairwise_iou > 0.20:
                    # Overlapping pieces detected — push IoU threshold above the max
                    best_iou = round(min(0.95, max_pairwise_iou + 0.10), 2)
                else:
                    # Well-separated pieces — standard threshold is fine
                    best_iou = 0.50

                info = (f"Conf={best_conf:.2f}  IoU={best_iou:.2f}  "
                        f"({n_kept} pieces, max overlap={max_pairwise_iou:.2f})")
                self.after(0, lambda: self._apply_tuned_thresholds(
                    best_conf, best_iou, info=info))
            except Exception as e:
                self.after(0, lambda: self._set_status(
                    f"Image analysis failed: {e}", color=ACCENT))
            finally:
                self.after(0, lambda: self._autotune_btn.config(
                    state="normal", text="⚡ Auto-Tune Thresholds"))

        threading.Thread(target=_analyze, daemon=True).start()

    # ─────────── Validation-set fallback ──────────────────────
    def _autotune_from_val(self):
        """Fall back to F1-optimal tuning on the validation split."""
        self._autotune_btn.config(state="disabled", text="⚡ Tuning (val set)...")
        self._set_status("Auto-tuning on validation set (no image loaded)...", color=TEXT_DIM)

        def _tune():
            try:
                if not DATA_YAML.exists():
                    raise FileNotFoundError(f"data.yaml not found at {DATA_YAML}")

                metrics = self._model.val(
                    data=str(DATA_YAML), split="val",
                    verbose=False, workers=0,
                )

                best_conf, best_iou = 0.25, 0.60

                if metrics and hasattr(metrics, "box") and metrics.box is not None:
                    if hasattr(metrics.box, "f1") and metrics.box.f1 is not None:
                        f1_curve = metrics.box.f1
                        if len(f1_curve.shape) > 1:
                            f1_curve = f1_curve[0]
                        best_idx  = int(np.argmax(f1_curve))
                        best_conf = best_idx / len(f1_curve)
                        best_conf = max(0.15, min(0.85, best_conf))

                self.after(0, lambda: self._apply_tuned_thresholds(
                    best_conf, best_iou,
                    info=f"F1-optimal Conf={best_conf:.2f} (val set)"))
            except Exception as e:
                self.after(0, lambda: self._set_status(
                    f"Tuning failed: {e}", color=ACCENT))
            finally:
                self.after(0, lambda: self._autotune_btn.config(
                    state="normal", text="⚡ Auto-Tune Thresholds"))

        threading.Thread(target=_tune, daemon=True).start()

    # ─────────── Apply tuned thresholds ───────────────────────
    def _apply_tuned_thresholds(self, conf: float, iou: float, info: str = ""):
        self._conf_var.set(round(conf, 2))
        self._iou_var.set(round(iou, 2))
        msg = f"✔ Auto-tuned: {info}" if info else f"✔ Auto-tuned: Conf={conf:.2f}  IoU={iou:.2f}"
        self._set_status(msg, color=TEXT_GREEN)
        self._rerun_if_image()

# ──────────────────── IoU helper ──────────────────────────────
def _compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format (x1,y1,x2,y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ──────────────────── Widget helpers ──────────────────────────
def _separator(parent):
    tk.Frame(parent, bg=BG_PANEL, height=1).pack(fill="x", padx=20, pady=8)


def _slider_row(parent, label: str, var: tk.DoubleVar,
                from_: float, to: float, on_change=None):
    row = tk.Frame(parent, bg=BG_CARD)
    row.pack(fill="x", padx=20, pady=3)

    tk.Label(row, text=label, font=(FONT_FAMILY, 9),
             bg=BG_CARD, fg=TEXT_LIGHT, width=14, anchor="w").pack(side="left")

    val_lbl = tk.Label(row, font=(FONT_FAMILY, 9, "bold"),
                       bg=BG_CARD, fg=ACCENT2, width=4)
    val_lbl.pack(side="right")

    def _update_label(*args):
        try:
            val_lbl.config(text=f"{var.get():.2f}")
        except tk.TclError:
            pass
        if on_change:
            on_change()

    # Trace variable updates so programmatic sets also update labels and trigger rerun
    var.trace_add("write", _update_label)
    val_lbl.config(text=f"{var.get():.2f}")

    tk.Scale(row, variable=var, from_=from_, to=to, resolution=0.01,
             orient="horizontal", bg=BG_CARD, fg=TEXT_LIGHT,
             troughcolor=BG_DARK, activebackground=ACCENT,
             highlightthickness=0, sliderrelief="flat",
             showvalue=False).pack(
                 side="left", fill="x", expand=True)


# ──────────────────── Entry point ─────────────────────────────
if __name__ == "__main__":
    app = ChickenDetectorApp()
    app.mainloop()
