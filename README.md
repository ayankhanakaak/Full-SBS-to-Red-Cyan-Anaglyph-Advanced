# Full SBS to Red/Cyan Anaglyph Advanced

A simple yet powerful PySide6 + OpenCV desktop application for converting full-width Side-by-Side (SBS) stereo images or videos into red/cyan anaglyphs. Designed for efficiency, flexibility, and user control — with GPU acceleration, frame-level keyframing, and modern GUI.

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/50b75746-4052-4304-967a-1ff4fc8fa308" />

---

## Features

- **Convert Full Width SBS images & videos** to red/cyan anaglyph (classic 3D glasses effect)
- **Keyframing:** Set focus/desaturation/mode for individual frames (videos)
- **Different modes:** Supports Color, Half-color, and Gray anaglyph modes
- **Frame interpolation:** Linear or stepped parameter transitions
- **Hardware acceleration:** Uses OpenCL (via OpenCV UMat) if available
- **Multithreaded video export** with progress dialog and cancel support
- **Batch image/video export, works with common formats**
- **Intuitive PySide6 GUI** with keyboard shortcuts for speedy workflow
- **No audio in exported videos (yet)**
- **Cross-platform (Windows, Linux, MacOS) with Python 3.7+**

---

## Requirements

- Python 3.7 or higher
- [PySide6](https://pypi.org/project/PySide6/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [numpy](https://pypi.org/project/numpy/)

Install all dependencies with:

```
pip install PySide6 opencv-python numpy
```

---

## Usage

1. **Clone or download** this repository.
2. **Run the application:**

   ```
   python Full-SBS-to-Red-Cyan-Anaglyph-Advanced-V.22.08.2025-1.py
   ```

3. **Open a full-width SBS image or video**
   - Supports: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.mpg`, `.mpeg`

4. **Adjust settings:**
   - Switch anaglyph mode (Color, Half-color, Gray)
   - Modify focus and desaturation
   - Use keyframing for complex video exports

5. **Export:**
   - Export an image as `.png`
   - Export a video as `.mp4` (no audio)

---

## Keyboard Shortcuts

- **Space:** Play/Pause video
- **Page Up / Page Down:** Previous/Next frame
- **Left / Right:** Decrease/Increase focus
- **Up / Down:** Decrease/Increase desaturation
- **Ctrl + O:** Open file
- **Ctrl + E:** Export
- **K:** Add/Update keyframe
- **Shift + K:** Delete keyframe
- **Ctrl + Shift + K:** Clear all keyframes

(See app UI for more.)

---

## Building a Standalone EXE

For Windows users, you can create a single-file executable with:

```
pyinstaller --onefile --windowed --icon="your_icon.ico" --add-data "your_icon.ico;." Full-SBS-to-Red-Cyan-Anaglyph-Advanced-V.22.08.2025-1.py
```

See [PyInstaller documentation](https://pyinstaller.org/) for details.

---

## Screenshot

Feel free to add a screenshot here for better visibility.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Author

Developed by [Your Name or Alias]  
[Your Website or Contact Link, if desired]

---

*Pull requests and feedback are welcome! Enjoy 3D!*
