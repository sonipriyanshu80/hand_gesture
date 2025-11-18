# Hand Gesture Recognition with OpenCV & MediaPipe

A real-time Python application that detects hand landmarks, draws MediaPipe-style skeleton visualization, and classifies hand gestures using your webcam.

## ✨ Features

- **Real-time hand tracking** using OpenCV and CVZone
- **21 hand landmarks** with MediaPipe-style visualization:
  - White skeleton lines connecting joints
  - Red dots for regular landmarks
  - Yellow dots for fingertips with ID labels
- **Fingertip coordinates** displayed as `(x, y)` below each tip
- **Gesture recognition** for:
  - 👊 Fist (0 fingers)
  - 👆 Pointing (1 finger)
  - 👍 Thumbs Up
  - ✌️ Victory (2 fingers)
  - 🖐️ Open Palm (5 fingers)
  - Custom finger counts (1-5)
- **Live FPS counter**
- **Translucent HUD** showing current gesture and finger count
- **Purple bounding box** and center point visualization

## 🔧 Requirements

- **Python 3.7 – 3.11** (⚠️ MediaPipe doesn't support Python 3.12+)
- Webcam
- Good lighting conditions

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd Hand_Gesture_Recognition
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install opencv-python cvzone mediapipe numpy
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. **Run the application:**
   ```bash
   python main.py
   ```

2. **Controls:**
   - Show your hand in front of the camera
   - Press `q` to quit

3. **What you'll see:**
   - Hand skeleton with white connecting lines
   - Red dots on joints, yellow dots on fingertips with ID labels
   - Coordinate labels below fingertips
   - Purple bounding box around detected hand
   - Translucent black HUD with gesture info
   - FPS counter in the top-right corner

## 📁 Project Structure

```
Hand_Gesture_Recognition/
├── main.py              # Main application file
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── .gitignore          # Git ignore rules
```

## 🔍 How It Works

1. **Hand Detection**: Uses CVZone's HandDetector to identify hand landmarks
2. **Landmark Visualization**: Draws 21 landmarks with connecting skeleton lines
3. **Finger Tracking**: Monitors fingertip positions (landmarks 4, 8, 12, 16, 20)
4. **Gesture Classification**: Analyzes finger states to identify common gestures
5. **Real-time Display**: Overlays all information on the live camera feed

## 🎯 Gesture Recognition

| Gesture | Description | Finger Pattern |
|---------|-------------|----------------|
| Fist | All fingers down | 0 fingers up |
| Pointing | Index finger extended | 1 finger up |
| Thumbs Up | Only thumb extended | Thumb only |
| Victory | Index + middle fingers | 2 specific fingers |
| Open Palm | All fingers extended | 5 fingers up |

## ⚙️ Configuration

Key parameters in `main.py`:
- `detectionCon=0.85`: Hand detection confidence threshold
- `maxHands=1`: Maximum number of hands to detect
- Camera resolution: 1280x720 (adjustable)

## 🔧 Troubleshooting

- **No hand detected**: Ensure good lighting and keep hand fully visible
- **Inaccurate gestures**: Adjust `detectionCon` value or improve lighting
- **Low FPS**: Reduce camera resolution or close other applications
- **Installation issues**: Verify Python version compatibility (3.7-3.11)

## 📋 Dependencies

```
opencv-python>=4.5.0
cvzone>=1.5.0
mediapipe>=0.8.0
numpy>=1.21.0
```

## 👨‍💻 Author 
5th Semester Student Project

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking technology
- [CVZone](https://github.com/cvzone/cvzone) for simplified computer vision
- [OpenCV](https://opencv.org/) for image processing capabilities
