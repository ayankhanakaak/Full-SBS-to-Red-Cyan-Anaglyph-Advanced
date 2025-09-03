# Full SBS to Red/Cyan Anaglyph Advanced

A simple yet powerful PySide6 + OpenCV desktop application for converting full-width Side-by-Side (SBS) stereo images or videos into red/cyan anaglyphs. Designed for efficiency, flexibility, and user control — with GPU acceleration, frame-level keyframing, and modern GUI.

<img width="1366" height="768" alt="Screenshot 2025-09-03 130034" src="https://github.com/user-attachments/assets/24e4eb1c-600b-4026-ac1c-a0b0ed4b9f0d" />

---

## Features

- **Convert Full Width SBS images & videos** to red/cyan anaglyph (classic 3D glasses effect)
- **Keyframing:** Set focus/desaturation/mode for individual frames (videos)
- **Different modes:** Supports Color, Half-color, and Gray anaglyph modes
- **Frame interpolation:** Linear or stepped parameter transitions
- **Hardware acceleration:** Uses OpenCL (via OpenCV UMat) if available
- **Multithreaded video export** with progress dialog and cancel support
- **Intuitive PySide6 GUI** with keyboard shortcuts for speedy workflow
- **Multiple FPS Methods:** Custom FPS, Frames and Duration Based FPS and OpenCV Based FPS
- **No audio in exported videos (yet)**

---

## Requirements

- [Python](https://www.python.org/downloads/) (Recommended Version [3.13.5](https://www.python.org/downloads/release/python-3135/))
- [PySide6](https://pypi.org/project/PySide6/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [numpy](https://pypi.org/project/numpy/)

Install all dependencies with:

```
pip install PySide6 opencv-python numpy
```
🟢If you download **.EXE** file (for Windows), no need to install any requirement.  
Just download and run.

---

## Usage

1. **Clone or download** this repository.
2. **Move inside the directory** where `Full SBS to Red-Cyan Anaglyph Advanced - V.22.08.2025-1.py` is located.
3. **Run the application:**

   ```
   python "Full SBS to Red-Cyan Anaglyph Advanced - V.22.08.2025-1.py"
   ```

4. **Open a full-width SBS image or video**
   - Supports: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.mpg`, `.mpeg`

5. **Adjust settings:**
   - Switch anaglyph mode (Color, Half-color, Gray)
   - Modify focus and desaturation
   - Use keyframing for complex video exports

6. **Export:**
   - Export an image as `.png`
   - Export a video as `.mp4` (no audio)

---

## Keyboard Shortcuts

- **Space:** Play/Pause video
- **Page Up / Page Down:** Previous/Next frame
- **Left / Right:** Decrease/Increase focus by 1px
- **Shift + Left/Right:** Decrease/Increse focus by 5px
- **Up / Down:** Decrease/Increase desaturation
- **Home:** Move at first frame
- **End:** Move at last frame
- **Ctrl + O:** Open file
- **Ctrl + E:** Export
- **K:** Add/Update keyframe
- **Shift + K:** Delete keyframe
- **Ctrl + Shift + K:** Clear all keyframes

---

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.

---

## Author

Developed by Ayan Khan  

---

*Feedbacks are welcome! Enjoy 3D!*
