#Simple PySide6 + OpenCV app for converting full-width SBS to red/cyan anaglyph.
#Version: 22.08.2025-1
#Install: pip install PySide6 opencv-python numpy

import sys
import os
import cv2
import numpy as np
import threading
import queue
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):  # PyInstaller temp folder
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
icon_path = resource_path("Full SBS to Anaglyph - Icon.ico")

# --- KEYFRAMING HELPERS (NEW) ---
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def interpolate_linear(v0, v1, t):
    return v0 + (v1 - v0) * t

def find_neighbor_keyframes(sorted_frames, idx):
    """Return (f_prev, f_next) where either may be None."""
    if not sorted_frames:
        return (None, None)
    # If idx is exactly a keyframe
    if idx in sorted_frames:
        return (idx, idx)
    # Find insertion point
    import bisect
    pos = bisect.bisect_left(sorted_frames, idx)
    f_prev = sorted_frames[pos - 1] if pos > 0 else None
    f_next = sorted_frames[pos] if pos < len(sorted_frames) else None
    return (f_prev, f_next)

def interp_params_at_frame(idx, keyframes_map, defaults, interp_mode):
    """
    keyframes_map: {frame_idx: {'focus': int, 'desat': int, 'mode': str}}
    defaults: {'focus': int, 'desat': int, 'mode': str}
    interp_mode: 'Linear' or 'Step'
    """
    if not keyframes_map:
        return defaults.copy()

    frames = sorted(keyframes_map.keys())
    f_prev, f_next = find_neighbor_keyframes(frames, idx)

    # Exact keyframe hit
    if f_prev is not None and f_prev == f_next:
        return keyframes_map[f_prev].copy()

    # Only previous exists (tail)
    if f_next is None and f_prev is not None:
        return keyframes_map[f_prev].copy()

    # Only next exists (head)
    if f_prev is None and f_next is not None:
        return keyframes_map[f_next].copy() if interp_mode == 'Step' else {
            'focus': keyframes_map[f_next]['focus'],
            'desat': keyframes_map[f_next]['desat'],
            'mode': keyframes_map[f_next]['mode'],
        }

    # Both exist: interpolate or step
    p0 = keyframes_map[f_prev]
    p1 = keyframes_map[f_next]
    if f_next == f_prev:
        return p0.copy()

    if interp_mode == 'Step':
        # Hold previous until next
        return p0.copy()

    # Linear interpolation for focus/desat; mode holds from prev
    span = max(1, (f_next - f_prev))
    t = (idx - f_prev) / float(span)
    focus = int(round(interpolate_linear(p0['focus'], p1['focus'], t)))
    desat = int(round(interpolate_linear(p0['desat'], p1['desat'], t)))
    mode = p0['mode']
    return {'focus': focus, 'desat': desat, 'mode': mode}
# --- END KEYFRAMING HELPERS ---

def anaglyph_umat(left_np: np.ndarray,
                  right_np: np.ndarray,
                  focus_px: int,
                  mode: str,
                  desat_percent: int) -> np.ndarray:
    """Return BGR np.ndarray. Uses OpenCL (UMat) when available; falls back to CPU automatically."""
    # Ensure OpenCL is enabled if available
    try:
        cv2.ocl.setUseOpenCL(True)
    except Exception:
        pass

    h, w = left_np.shape[:2]

    # Upload to UMat
    L = cv2.UMat(left_np)
    R = cv2.UMat(right_np)

    # Resize right to match left
    R = cv2.resize(R, (w, h), interpolation=cv2.INTER_AREA)

    # Symmetric shifts
    half_sep = int(focus_px) // 2
    shift_l = +half_sep
    shift_r = -half_sep

    M_L = np.float32([[1, 0, shift_l], [0, 1, 0]])
    M_R = np.float32([[1, 0, shift_r], [0, 1, 0]])

    Ls = cv2.warpAffine(L, M_L, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    Rs = cv2.warpAffine(R, M_R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Compose according to mode while staying in UMat
    if mode == "Color":
        bR, gR, rR = cv2.split(Rs)
        bL, gL, rL = cv2.split(Ls)
        out = cv2.merge([bR, gR, rL])
    elif mode == "Half-color":
        grayL = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY)
        bR, gR, _ = cv2.split(Rs)
        out = cv2.merge([bR, gR, grayL])
    else:  # "Gray"
        grayL = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
        out = cv2.merge([grayR, grayR, grayL])

    # Desaturation in HSV if requested
    desat = max(0, min(100, int(desat_percent))) / 100.0
    if desat > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        s_f32 = cv2.UMat(np.float32(s_ch.get()))  # operate in float then back to 8u
        s_f32 = s_f32 * (1.0 - desat)
        s_ch = cv2.UMat(np.uint8(np.clip(s_f32.get(), 0, 255)))
        hsv = cv2.merge([h_ch, s_ch, v_ch])
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out.get()

cv2.setNumThreads(cv2.getNumberOfCPUs())  #all logical cores
def bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()

def split_full_sbs(frame: np.ndarray):
    h, w = frame.shape[:2]
    half = w // 2
    left = frame[:, :half]
    right = frame[:, half:]
    return left, right

def apply_shift(img: np.ndarray, shift_px: int, use_ocl=False):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, shift_px], [0, 1, 0]])
    if use_ocl:
        try:
            uimg = cv2.UMat(img)
            out = cv2.warpAffine(uimg, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return out.get()
        except Exception:
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def make_anaglyph(left: np.ndarray,
                  right: np.ndarray,
                  focus_px: int = 10,
                  mode: str = "Color",
                  desat_percent: int = 0,
                  use_ocl: bool = False) -> np.ndarray:

    h, w = left.shape[:2]
    if use_ocl:
        right = cv2.resize(cv2.UMat(right), (w, h), interpolation=cv2.INTER_AREA).get()
    else:
        right = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)

    half_sep = focus_px // 2
    shift_l = +half_sep
    shift_r = -half_sep

    left_s = apply_shift(left, shift_l, use_ocl)
    right_s = apply_shift(right, shift_r, use_ocl)

    if mode == "Color":
        out = np.zeros_like(left_s)
        out[:, :, 2] = left_s[:, :, 2]
        out[:, :, 1] = right_s[:, :, 1]
        out[:, :, 0] = right_s[:, :, 0]
    elif mode == "Half-color":
        grayL = cv2.cvtColor(left_s, cv2.COLOR_BGR2GRAY)
        out = np.zeros_like(left_s)
        out[:, :, 2] = grayL
        out[:, :, 1] = right_s[:, :, 1]
        out[:, :, 0] = right_s[:, :, 0]
    else:
        grayL = cv2.cvtColor(left_s, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_s, cv2.COLOR_BGR2GRAY)
        out = cv2.merge([grayR, grayR, grayL])

    desat = np.clip(desat_percent, 0, 100) / 100.0
    if desat > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        s_ch = (s_ch.astype(np.float32) * (1.0 - desat)).clip(0, 255).astype(np.uint8)
        hsv = cv2.merge([h_ch, s_ch, v_ch])
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out

class VideoPlayer(QtWidgets.QMainWindow):    
    def set_ui_enabled(self, enabled: bool):
        """Enable/disable all interactive UI controls except the progress dialog."""
        self.open_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)
        self.play_btn.setEnabled(enabled)
        self.prev_btn.setEnabled(enabled)
        self.next_btn.setEnabled(enabled)
        self.pos_slider.setEnabled(enabled)
        self.focus_slider.setEnabled(enabled)
        self.desat_slider.setEnabled(enabled)
        self.mode_combo.setEnabled(enabled)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Full SBS → Red/Cyan Anaglyph")
        self.resize(1100, 700)

        #State
        self.keyframes = {}  # frame_idx -> {'focus': int, 'desat': int, 'mode': str}
        self.interp_mode = 'Linear'
        self.cap = None
        self.is_video = False
        self.image_frame = None  #for images
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0

        #Controls
        self.video_label = QtWidgets.QLabel(alignment=Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        self.video_label.setMinimumSize(400, 300)

        #Top bar
        self.open_btn = QtWidgets.QPushButton("Open")
        self.open_btn.clicked.connect(self.open_file)
        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.clicked.connect(self.export_output)
        self.export_btn.setEnabled(False)

        #Playback controls
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn = QtWidgets.QPushButton("⟨")
        self.prev_btn.clicked.connect(lambda: self.step_frames(-1))
        self.next_btn = QtWidgets.QPushButton("⟩")
        self.next_btn.clicked.connect(lambda: self.step_frames(1))

        #Timeline
        self.time_label = QtWidgets.QLabel("00:00 / 00:00")
        self.frame_label = QtWidgets.QLabel("Frame: 0")
        self.pos_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.pos_slider.setRange(0, 0)
        self.pos_slider.sliderPressed.connect(self.pause_for_seek)
        self.pos_slider.sliderReleased.connect(self.seek_release)
        self.pos_slider.valueChanged.connect(self.seek_changed)

        #Anaglyph controls
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Color", "Half-color", "Gray"])

        self.focus_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.focus_slider.setRange(-200, 200)
        self.focus_slider.setValue(10)
        self.focus_label = QtWidgets.QLabel("Focus: 10 px")

        self.desat_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.desat_slider.setRange(0, 100)
        self.desat_slider.setValue(0)
        self.desat_label = QtWidgets.QLabel("Desaturation: 0%")

        for s in (self.focus_slider, self.desat_slider):
            s.setSingleStep(1)
            s.setPageStep(5)

        self.focus_slider.valueChanged.connect(self.update_labels_and_refresh)
        self.desat_slider.valueChanged.connect(self.update_labels_and_refresh)
        self.mode_combo.currentIndexChanged.connect(self.refresh_frame)
        
        self.kf_add_btn = QtWidgets.QPushButton("Add/Update KF")
        self.kf_add_btn.clicked.connect(self.add_or_update_keyframe)

        self.kf_del_btn = QtWidgets.QPushButton("Delete KF")
        self.kf_del_btn.clicked.connect(self.delete_keyframe_at_current)

        self.kf_clear_btn = QtWidgets.QPushButton("Clear KFs")
        self.kf_clear_btn.clicked.connect(self.clear_keyframes)

        self.interp_combo = QtWidgets.QComboBox()
        self.interp_combo.addItems(["Linear", "Step"])
        self.interp_combo.currentIndexChanged.connect(self.on_interp_changed)

        self.kf_list = QtWidgets.QListWidget()
        self.kf_list.setMaximumHeight(120)

        self.kf_status = QtWidgets.QLabel("")  # shows whether preview is using keyframed params

        #Layouts
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(self.open_btn)
        top_bar.addWidget(self.export_btn)
        top_bar.addStretch(1)

        play_bar = QtWidgets.QHBoxLayout()
        play_bar.addWidget(self.play_btn)
        play_bar.addWidget(self.prev_btn)
        play_bar.addWidget(self.next_btn)
        play_bar.addWidget(self.time_label)
        play_bar.addWidget(self.frame_label)
        play_bar.addWidget(self.pos_slider)

        ana_grid = QtWidgets.QGridLayout()
        row = 0
        ana_grid.addWidget(QtWidgets.QLabel("Mode:"), row, 0)
        ana_grid.addWidget(self.mode_combo, row, 1, 1, 1)
        row += 1
        ana_grid.addWidget(self.focus_label, row, 0)
        ana_grid.addWidget(self.focus_slider, row, 1, 1, 3)
        row += 1
        ana_grid.addWidget(self.desat_label, row, 0)
        ana_grid.addWidget(self.desat_slider, row, 1, 1, 3)
        row += 1
        ana_grid.addWidget(QtWidgets.QLabel("Interpolation:"), row, 0)
        ana_grid.addWidget(self.interp_combo, row, 1)
        ana_grid.addWidget(self.kf_status, row, 0, 1, 4)
        ana_grid.addWidget(self.kf_add_btn, row, 2)
        ana_grid.addWidget(self.kf_del_btn, row, 3)
        ana_grid.addWidget(self.kf_clear_btn, row, 4)
        row += 1
        ana_grid.addWidget(QtWidgets.QLabel("Keyframes:"), row, 0)
        ana_grid.addWidget(self.kf_list, row, 1, 1, 4)
        
        controls = QtWidgets.QVBoxLayout()
        controls.addLayout(top_bar)
        controls.addWidget(self.video_label, stretch=1)
        controls.addLayout(play_bar)
        controls.addLayout(ana_grid)

        central = QtWidgets.QWidget()
        central.setLayout(controls)
        self.setCentralWidget(central)

        #Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_play)
        QtGui.QShortcut(QtGui.QKeySequence("Page Up"), self, activated=lambda: self.step_frames(-1))
        QtGui.QShortcut(QtGui.QKeySequence("Page Down"), self, activated=lambda: self.step_frames(1))
        QtGui.QShortcut(QtGui.QKeySequence("Left"), self, activated=lambda: self.bump_focus(-1))
        QtGui.QShortcut(QtGui.QKeySequence("Right"), self, activated=lambda: self.bump_focus(+1))
        QtGui.QShortcut(QtGui.QKeySequence("Shift+Left"), self, activated=lambda: self.bump_focus(-5))
        QtGui.QShortcut(QtGui.QKeySequence("Shift+Right"), self, activated=lambda: self.bump_focus(+5))
        QtGui.QShortcut(QtGui.QKeySequence("Up"), self, activated=lambda: self.bump_desat(-5))
        QtGui.QShortcut(QtGui.QKeySequence("Down"), self, activated=lambda: self.bump_desat(+5))
        QtGui.QShortcut(QtGui.QKeySequence("Home"), self, activated=lambda: self.seek_to(0))
        QtGui.QShortcut(QtGui.QKeySequence("End"), self, activated=self.seek_to_end)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl + O"), self, activated=self.open_file)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl + E"), self, activated=self.export_output)
        QtGui.QShortcut(QtGui.QKeySequence("K"), self, activated=self.add_or_update_keyframe)
        QtGui.QShortcut(QtGui.QKeySequence("Shift+K"), self, activated=self.delete_keyframe_at_current)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+K"), self, activated=self.clear_keyframes)

        self.update_labels_and_refresh()
        
        self.setWindowIcon(QtGui.QIcon('Full SBS to Anaglyph - Icon.ico'))

    # ------- Keyframing logic (NEW) -------
    def on_interp_changed(self):
        self.interp_mode = self.interp_combo.currentText()
        self.refresh_frame()

    def defaults_from_ui(self):
        return {
            'focus': int(self.focus_slider.value()),
            'desat': int(self.desat_slider.value()),
            'mode': self.mode_combo.currentText()
        }

    def add_or_update_keyframe(self):
        idx = self.current_frame_idx if self.is_video else 0
        params = self.defaults_from_ui()
        self.keyframes[idx] = params
        self.refresh_kf_list()
        self.refresh_frame()

    def delete_keyframe_at_current(self):
        idx = self.current_frame_idx if self.is_video else 0
        if idx in self.keyframes:
            del self.keyframes[idx]
            self.refresh_kf_list()
            self.refresh_frame()

    def clear_keyframes(self):
        self.keyframes.clear()
        self.refresh_kf_list()
        self.refresh_frame()

    def refresh_kf_list(self):
        self.kf_list.clear()
        for f in sorted(self.keyframes.keys()):
            p = self.keyframes[f]
            self.kf_list.addItem(f"Frame {f}: focus={p['focus']} px, desat={p['desat']}%, mode={p['mode']}")

    def params_for_frame(self, frame_idx):
        defaults = self.defaults_from_ui()
        p = interp_params_at_frame(frame_idx, self.keyframes, defaults, self.interp_mode)
        # Clamp to UI ranges
        p['focus'] = clamp(p['focus'], self.focus_slider.minimum(), self.focus_slider.maximum())
        p['desat'] = clamp(p['desat'], self.desat_slider.minimum(), self.desat_slider.maximum())
        if p['mode'] not in ["Color", "Half-color", "Gray"]:
            p['mode'] = defaults['mode']
        return p

    def update_kf_status(self, used_params):
        d = self.defaults_from_ui()
        is_kf = (used_params['focus'] != d['focus'] or
                 used_params['desat'] != d['desat'] or
                 used_params['mode'] != d['mode'])
        if self.keyframes:
            if is_kf:
                self.kf_status.setText(f"Using keyframed params → focus={used_params['focus']} px, desat={used_params['desat']}%, mode={used_params['mode']} [{self.interp_mode}]")
            else:
                self.kf_status.setText("Using default UI params (no KF at this frame)")
        else:
            self.kf_status.setText("No keyframes")
    # ------- End keyframing logic -------

    #------- UI helpers -------
    def update_labels_and_refresh(self):
        self.focus_label.setText(f"Focus: {self.focus_slider.value()} px")
        self.desat_label.setText(f"Desaturation: {self.desat_slider.value()}%")
        self.frame_label.setText(f"Frame: {self.current_frame_idx}")
        # Optionally live‑inject values into current frame's params
        if self.is_video:
            self.keyframes[self.current_frame_idx] = self.defaults_from_ui()
        self.refresh_frame()

    def bump_focus(self, delta):
        self.focus_slider.setValue(int(np.clip(self.focus_slider.value() + delta, self.focus_slider.minimum(), self.focus_slider.maximum())))

    def bump_desat(self, delta):
        self.desat_slider.setValue(int(np.clip(self.desat_slider.value() + delta, self.desat_slider.minimum(), self.desat_slider.maximum())))

    #------- File I/O -------
    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Full SBS Video or Image", "", 
                                                        "Media Files (*.mp4 *.mkv *.mov *.avi *.webm *.mpg *.mpeg *.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not path:
            return
        self.stop()

        #Try image first
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None and img.size > 0 and img.shape[1] % 2 == 0:
            self.is_video = False
            self.image_frame = img
            self.cap = None
            self.total_frames = 1
            self.fps = 1.0
            self.current_frame_idx = 0
            self.pos_slider.setRange(0, 0)
            self.export_btn.setEnabled(True)
            self.refresh_frame()
            return

        #Otherwise video
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to open media.")
            self.cap = None
            return

        #Validate first frame shape
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.shape[1] % 2 != 0:
            QtWidgets.QMessageBox.critical(self, "Error", "This doesn't look like full-width SBS (even width required).")
            self.cap.release()
            self.cap = None
            return

        self.is_video = True
        self.image_frame = None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.current_frame_idx = 0
        self.pos_slider.setRange(0, max(0, self.total_frames - 1))
        self.export_btn.setEnabled(True)
        self.refresh_frame()

    #------- Playback -------
    def toggle_play(self):
        if not self.is_video or self.cap is None:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("Pause")
            interval_ms = max(1, int(1000.0 / max(1e-3, self.fps)))
            self.timer.start(interval_ms)
        else:
            self.play_btn.setText("Play")
            self.timer.stop()

    def stop(self):
        self.playing = False
        self.timer.stop()
        self.play_btn.setText("Play")

    def next_frame(self):
        if not self.is_video or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return
        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.display_anaglyph(frame)
        #Update slider and time
        if self.total_frames > 1:
            self.pos_slider.blockSignals(True)
            self.pos_slider.setValue(self.current_frame_idx)
            self.pos_slider.blockSignals(False)
        self.update_time_label()

    def step_frames(self, delta):
        if not self.is_video or self.cap is None:
            return
        self.stop()
        new_idx = int(np.clip(self.current_frame_idx + delta, 0, max(0, self.total_frames - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = new_idx
            self.display_anaglyph(frame)
            self.pos_slider.blockSignals(True)
            self.pos_slider.setValue(self.current_frame_idx)
            self.pos_slider.blockSignals(False)
            self.update_time_label()

    def pause_for_seek(self):
        self.was_playing = self.playing
        self.stop()

    def seek_release(self):
        if getattr(self, "was_playing", False):
            self.toggle_play()

    def seek_changed(self, value):
        if not self.is_video or self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = int(value)
            self.display_anaglyph(frame)
            self.update_time_label()

    def seek_to(self, frame_idx):
        if not self.is_video or self.cap is None:
            return
        self.stop()
        idx = int(np.clip(frame_idx, 0, max(0, self.total_frames - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = idx
            self.display_anaglyph(frame)
            self.pos_slider.blockSignals(True)
            self.pos_slider.setValue(self.current_frame_idx)
            self.pos_slider.blockSignals(False)
            self.update_time_label()

    def seek_to_end(self):
        if self.is_video and self.total_frames > 0:
            self.seek_to(self.total_frames - 1)

    def update_time_label(self):
        if self.is_video and self.fps > 0:
            cur_s = self.current_frame_idx / self.fps
            tot_s = (self.total_frames - 1) / self.fps if self.total_frames > 1 else cur_s
        else:
            cur_s = tot_s = 0

        def fmt(t):
            m = int(t // 60)
            s = int(t % 60)
            return f"{m:02d}:{s:02d}"

        self.time_label.setText(f"{fmt(cur_s)} / {fmt(tot_s)}")
        self.frame_label.setText(f"Frame: {self.current_frame_idx}")

    #------- Rendering -------
    def refresh_frame(self):
        if self.is_video and self.cap is not None:
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            #Try to fetch current frame without advancing timeline
            if pos != self.current_frame_idx:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                #Reset back so play continues correctly
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                self.display_anaglyph(frame)
                self.update_time_label()
        elif self.image_frame is not None:
            self.display_anaglyph(self.image_frame)

    def display_anaglyph(self, frame_bgr: np.ndarray):
        if frame_bgr is None or frame_bgr.size == 0:
            return
        if frame_bgr.shape[1] % 2 != 0:
            return  # not SBS

        left, right = split_full_sbs(frame_bgr)

        frame_idx = self.current_frame_idx if self.is_video else 0
        params = self.params_for_frame(frame_idx)
        self.update_kf_status(params)

        out = make_anaglyph(
            left, right,
            focus_px=int(params['focus']),
            mode=params['mode'],
            desat_percent=int(params['desat']),
            use_ocl=True
        )
        self.show_image(out)

    def show_image(self, img_bgr: np.ndarray):
        qimg = bgr_to_qimage(img_bgr)
        pix = QtGui.QPixmap.fromImage(qimg)
        #Fit to label while keeping aspect ratio
        scaled = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        #Redraw current content to fit new size
        self.refresh_frame()

    #------- Export -------
    def export_output(self):
        # Single-image export (unchanged)
        if self.image_frame is not None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save image", "anaglyph.png", "PNG (*.png)")
            if not path:
                return
            left, right = split_full_sbs(self.image_frame)
            out = anaglyph_umat(
                left_np=left,
                right_np=right,
                focus_px=int(self.focus_slider.value()),
                mode=self.mode_combo.currentText(),
                desat_percent=int(self.desat_slider.value()),
            )
            cv2.imwrite(path, out)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved image: {os.path.basename(path)}")
            return

        # Video export
        if not self.is_video or self.cap is None:
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export video", "anaglyph.mp4", "MP4 (*.mp4)")
        if not path:
            return

        # Peek first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame0 = self.cap.read()
        if not ret:
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot read video frames.")
            return

        # Fixed params for consistent output
        focus_px = int(self.focus_slider.value())
        mode = self.mode_combo.currentText()
        desat_percent = int(self.desat_slider.value())
        
        kf_snapshot = {k: v.copy() for k, v in self.keyframes.items()}
        interp_mode = self.interp_combo.currentText()
        defaults = {
            'focus': focus_px,
            'desat': desat_percent,
            'mode': mode
        }

        left0, right0 = split_full_sbs(frame0)
        sample_out = anaglyph_umat(left0, right0, focus_px, mode, desat_percent)
        h, w = sample_out.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if not writer.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to open VideoWriter.")
            return

        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        # Progress dialog
        dlg = QtWidgets.QProgressDialog("Exporting...", "Cancel", 0, total, self)
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        # Freeze UI
        self.set_ui_enabled(False)

        # Queues and control
        QN = 8  # queue size (tune this)
        q_dec = queue.Queue(maxsize=QN)      # raw frames (numpy BGR)
        q_proc = queue.Queue(maxsize=QN)     # processed anaglyph frames (numpy BGR)
        

        stop = threading.Event()
        canceled = threading.Event()

        # Stage 1: Decode (CPU)
        def t_decode():
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                idx = 0
                while not stop.is_set():
                    if canceled.is_set():
                        break
                    ret, frm = self.cap.read()
                    if not ret:
                        break
                    try:
                        q_dec.put((idx, frm), timeout=0.5)
                    except queue.Full:
                        if stop.is_set() or canceled.is_set():
                            break
                        continue
                    idx += 1
            finally:
                # Signal end of stream
                try:
                    q_dec.put(None, timeout=0.5)
                except Exception:
                    pass

        # Stage 2: Process (iGPU via UMat)
        def t_process():
            try:
                while not stop.is_set():
                    if canceled.is_set():
                        break
                    data = q_dec.get()
                    if data is None:
                        break
                    idx, frm = q_dec.get()
                    if idx is None:
                        break
                    left, right = split_full_sbs(frm)
                    # Get parameters for this frame from snapshot
                    params = interp_params_at_frame(idx, kf_snapshot, defaults, interp_mode)
                    params['focus'] = int(clamp(params['focus'], -200, 200))
                    params['desat'] = int(clamp(params['desat'], 0, 100))
                    if params['mode'] not in ["Color", "Half-color", "Gray"]:
                        params['mode'] = defaults['mode']
                    out = anaglyph_umat(left, right,
                                        params['focus'],
                                        params['mode'],
                                        params['desat'])
                    while not stop.is_set():
                        try:
                            q_proc.put(out, timeout=0.5)
                            break
                        except queue.Full:
                            if canceled.is_set():
                                return
                            continue
            finally:
                try:
                    q_proc.put(None, timeout=0.5)
                except Exception:
                    pass

        # Stage 3: Encode (CPU write)
        progress = {"n": 0}
        def t_encode():
            try:
                while not stop.is_set():
                    if canceled.is_set():
                        break
                    out = q_proc.get()
                    if out is None:
                        break
                    writer.write(out)
                    progress["n"] += 1
            finally:
                pass

        # Threads
        th_dec = threading.Thread(target=t_decode, daemon=True)
        th_prc = threading.Thread(target=t_process, daemon=True)
        th_enc = threading.Thread(target=t_encode, daemon=True)

        th_dec.start(); th_prc.start(); th_enc.start()

        # UI loop to handle progress + cancel
        try:
            while th_enc.is_alive():
                if dlg.wasCanceled():
                    canceled.set(); stop.set()
                    break
                # update progress occasionally
                if total > 0:
                    dlg.setValue(min(progress["n"], total))
                QtWidgets.QApplication.processEvents()
                QtCore.QThread.msleep(10)
        finally:
            # ensure shutdown
            stop.set()
            th_dec.join(timeout=1.0)
            th_prc.join(timeout=1.0)
            th_enc.join(timeout=1.0)
            try:
                writer.release()
            except Exception:
                pass
            try:
                dlg.close()
            except Exception:
                pass
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            self.set_ui_enabled(True)
            self.refresh_frame()

        if canceled.is_set():
            QtWidgets.QMessageBox.information(self, "Canceled", "Export canceled. Partial file may be unusable.")
        else:
            QtWidgets.QMessageBox.information(self, "Done", f"Exported: {os.path.basename(path)} (no audio)")


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(icon_path))
    win = VideoPlayer()
    win.setWindowIcon(QtGui.QIcon(icon_path))
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()