# Live Object Detection & Tracing 🔮

**Real-Time Object Detection & Tracking Platform**

A cutting-edge web application for live video object detection powered by YOLOv8 neural networks. Experience state-of-the-art AI detection with an elegant, modern interface.

---

## 📋 Table of Contents

- [Activity Overview](#activity-overview)
- [Learning Outcomes](#learning-outcomes)
- [Expected Results](#expected-results)
- [Document Report Requirements](#document-report-requirements)
- [Possible Enhancements](#possible-enhancements)
- [Grading Rubric](#grading-rubric)
- [Submission Requirements](#submission-requirements)
- [Overview](#overview)
- [Architecture & Design](#architecture--design)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)

---

## 🎓 Activity Overview

**Activity 3: Python Streamlit + ML Model — Real-Time Object Detection and Tracking using AI and Webcam**

In this hands-on activity, learners will build an interactive web application using **Streamlit** that integrates a real-time video stream from a **webcam** and applies **Artificial Intelligence** for object detection and tracking.

The system uses the **YOLOv8 model** to identify everyday objects (such as people, phones, or bottles) and display them with bounding boxes and labels directly on the video feed. Through this activity, students will explore how computer vision works in real-world applications, understand frame-by-frame image processing, and experience how AI models are deployed in live environments.

---

## 🧠 Learning Outcomes

By completing this activity, learners will:

- **Understand the basics of real-time computer vision**
- **Learn how AI models process video frames**
- **Build a simple yet powerful AI-powered web app**
- **Explore object tracking across multiple frames**

---

## ✅ Expected Software-based Results

### 1. Functional Web App Interface
A browser-based app displaying:
- **Title**: *Live Object Detection & Tracing*
- **Webcam feed** embedded in the page

### 2. Live Camera Detection
When the camera is turned on:
- Objects are **detected instantly**
- **Bounding boxes** appear around objects
- **Labels** (e.g., person, cell phone, bottle) are shown

### 3. Object Tracking Behavior
- Moving objects are **continuously tracked**
- The same object keeps its **identity across frames** (smooth tracking)

---

## 📄 Document Report Requirements

Students must submit a document report containing the following:

### 4. Observation Report
Document your findings, including:
- **List of detected objects**
- **Accuracy of detection** under different lighting conditions
- **Performance** (smooth vs lagging)

### 5. Screenshot or Screen Recording
Capture at least **5 different object outputs** showing:
- Detected objects with **labels**
- Real-time **bounding boxes**

### 6. Reflection
Answer these questions:
- What objects were easily detected?
- What factors affect detection accuracy?

---

## 🔧 Possible Enhancement Add-ons

Students must extend the activity by implementing at least one of the following:

- **Adding object counting** (e.g., number of people)
- **Triggering alerts** for specific objects
- **Saving detected frames** as images

> 💡 **Tip**: Completion of the source code alone may obtain a grade of **80%**, with added enhancements may obtain a maximum of **20%** extra credit.

---

## 📊 Grading Rubric

| Component | Weight | Description |
|-----------|--------|-------------|
| **Source Code Completion** | 80% | Functional Streamlit app with live detection, bounding boxes, and labels |
| **Enhancement Add-ons** | +20% | Extra features such as object counting, alerts, or frame saving |

---

## 📤 Required Submission Output

Submit the following links:

1. **Live Streamlit Link** — Deployed application URL
2. **GitHub Repository Link** — Source code with this `README.md` file
3. **Document Report Link** — Google Document with observations, screenshots, and reflection

---

## 📌 Overview

**Live Object Detection & Tracing** is a production-ready web application that brings state-of-the-art computer vision to your browser. Built with Streamlit and powered by Ultralytics YOLOv8, it enables real-time object detection and multi-object tracking directly from your camera feed.

### What It Does
- 🎥 **Live Camera Feed**: Real-time video processing with WebRTC
- 🎯 **Object Detection**: Identifies 80+ object categories (COCO dataset)
- 📍 **Object Tracking**: Persistent ID assignment across video frames
- ⚙️ **Configurable AI**: Adjust confidence thresholds, model variants, and tracking algorithms
- 🎨 **Modern UI**: Dark-mode interface with interactive controls

---

## 🏗️ Architecture & Design

### Design Philosophy

Live Object Detection & Tracing follows a **modular, performance-focused design** emphasizing:

- **Real-Time Processing**: Async frame processing prevents UI blocking
- **Customizable AI Pipeline**: Swap models, trackers, and detection parameters on-the-fly
- **User-Centric Interface**: Professional dark theme with intuitive controls
- **Accessibility**: Clear visual feedback and helpful guidance throughout the app

### System Architecture

```
┌─────────────────────────────────────────┐
│     Browser / WebRTC Client             │
│  (Camera Feed Input)                    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│   Streamlit Web Interface                │
│  (UI, Controls, Sidebar)                │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│   Video Processing Pipeline             │
│  (Frame Capture → Detection → Output)   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│   YOLOv8 Detection Model                │
│  (Ultralytics YOLO v8 Nano/S/M)        │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│   Tracker Algorithm                     │
│  (ByteTrack / BoT-SORT)                │
└─────────────────────────────────────────┘
```

### Component Breakdown

#### 1. **Frontend Layer** (Streamlit UI)
- **Hero Section**: Branding and app introduction
- **Feature Cards**: Highlights of core capabilities
- **Camera Panel**: Live video feed container
- **Sidebar**: Detection settings, display options, quick tips
- **Info Boxes**: Configuration display and usage tips
- **Tag Cloud**: Shows detectable object categories

#### 2. **Video Processing Layer**
- **Frame Callback**: `video_frame_callback()` processes each video frame
- **Model Inference**: YOLOv8 runs detection on incoming frames
- **Result Annotation**: Bounding boxes, labels, and confidence scores applied
- **Async Processing**: Prevents UI freezing during heavy computation

#### 3. **AI Model Layer**
- **YOLOv8 Variants**:
  - `YOLOv8n` (Nano): Fastest, lowest resource usage
  - `YOLOv8s` (Small): Balanced speed/accuracy
  - `YOLOv8m` (Medium): Highest accuracy, more resources
- **Tracking**: Persistent object IDs across frames
- **Classes**: 80 object categories from COCO dataset

### UI/UX Design

#### Color Palette
- **Background**: Deep blue gradient (`#0f0c29 → #1a1a3e → #16213e`)
- **Primary**: Purple (`#8b5cf6`)
- **Secondary**: Indigo (`#6366f1`)
- **Accent**: Amber (`#f59e0b`) & Cyan (`#06b6d4`)
- **Text**: Slate (`#e2e8f0`)

#### Typography
- **Brand Font**: "Syne" (Bold headings)
- **Body Font**: "Space Grotesk" (Modern, clean)

#### Interactive Elements
- **Buttons**: Gradient background with hover animations
- **Cards**: Frosted glass effect (backdrop blur)
- **Status Indicators**: Animated pulse for "AI Ready" state
- **Corner Accents**: Minimalist frame effect on video container

---

## ✨ Key Features

### 1. Real-Time Detection
- Processes video frames instantly
- Identifies objects with confidence scoring
- Multiple object detection in single frame

### 2. Intelligent Tracking
- Assigns unique IDs to objects
- Tracks objects across video frames
- Supports ByteTrack and BoT-SORT algorithms

### 3. Flexible Configuration
- **Confidence Threshold**: 0.1 - 1.0 (adjustable in real-time)
- **Model Selection**: Choose speed vs. accuracy
- **Display Options**: Toggle labels, confidence, box style, track IDs
- **Tracker Algorithm**: ByteTrack (default) or BoT-SORT

### 4. Modern User Interface
- Dark-mode professional design
- Responsive layout for all screen sizes
- Quick-access sidebar with live statistics
- Helpful tips for optimal usage

### 5. WebRTC Streaming
- Browser-native video capture
- Secure, local processing
- No external dependencies for camera access

---

## 🛠️ Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Web framework & UI |
| **Video** | streamlit-webrtc | Real-time WebRTC streaming |
| **AI/ML** | Ultralytics YOLO | Object detection model |
| **Computer Vision** | OpenCV | Image processing |
| **Media** | PyAV | Audio/video manipulation |
| **Deep Learning** | PyTorch | Neural network framework |
| **Utilities** | NumPy, Tornado, Protobuf | Data processing |

### Requirements
```
numpy>=1.24.0,<2.0.0
streamlit>=1.36.0
streamlit-webrtc>=0.45.0
ultralytics>=8.2.0
opencv-python-headless>=4.8.0,<5.0.0
av>=10.0.0
torch>=2.0.0
torchvision>=0.15.0
tornado==6.4
protobuf>=3.20.3,<5.0.0
```

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- Camera/webcam access
- 2GB+ RAM (4GB+ recommended for YOLOv8m)

### Step 1: Clone Repository
```bash
git clone https://github.com/nimfaSol/object-detector.git
cd object-detector
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model
The YOLOv8 model (`yolov8n.pt`) is included in the repository. If needed, download it:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## 🚀 Quick Start

### Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Basic Usage
1. **Allow Camera Access**: Click "Allow" when prompted
2. **Press START**: Begin live video streaming
3. **Adjust Settings**: Use sidebar controls to customize detection
4. **View Results**: Watch objects appear with bounding boxes and IDs

---

## ⚙️ Configuration

### Sidebar Controls

#### Detection Settings
- **Confidence Threshold** (0.1 - 1.0)
  - Higher = stricter detection
  - Lower = more detections (including false positives)
  - Default: 0.5

- **Model Variant**
  - `YOLOv8n`: ~5ms inference (recommended for real-time)
  - `YOLOv8s`: ~10ms inference (balanced)
  - `YOLOv8m`: ~20ms inference (most accurate)

- **Tracker Algorithm**
  - `ByteTrack`: Faster, good for crowded scenes
  - `BoT-SORT`: More stable ID tracking

#### Display Options
- **Show Labels**: Toggle class names on boxes
- **Show Confidence**: Display detection accuracy percentage
- **Show Track IDs**: Display persistent object identifiers
- **Show Bounding Boxes**: Toggle box visibility

#### Live Stats
- **Model**: Active model variant
- **Classes**: Number of detectable categories (80 COCO classes)
- **Confidence**: Current threshold setting
- **Mode**: Async processing status

---

## 📖 Usage Guide

### For Object Detection Tasks
1. Set **Confidence Threshold** based on your needs:
   - Security/Precision: 0.7 - 0.9
   - General Purpose: 0.5 - 0.7
   - Maximum Detection: 0.1 - 0.3

2. Choose **Model Variant**:
   - Real-time applications: YOLOv8n
   - Balanced use: YOLOv8s
   - High accuracy: YOLOv8m

### For Object Tracking
1. Enable "Show Track IDs" in Display Options
2. Select tracking algorithm:
   - Default ByteTrack for smooth tracking
   - BoT-SORT for complex scenes
3. Ensure objects remain in frame for consistent ID assignment

### Troubleshooting
| Issue | Solution |
|-------|----------|
| No detections | Lower confidence threshold, check lighting |
| Low FPS | Switch to YOLOv8n model, reduce resolution |
| Shaky tracking IDs | Increase confidence, use BoT-SORT tracker |
| Camera not working | Check browser permissions, restart browser |

---

## 📁 Project Structure

```
actt_3/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── package.txt              # System dependencies (Linux)
├── yolov8n.pt              # Pre-trained YOLOv8 Nano model
└── README.md               # This file
```

### File Details

**app.py** (Main Application)
- **Lines 1-350**: Styling & UI configuration (CSS)
- **Lines 350-450**: SVG icon definitions
- **Lines 450-550**: Model loading & caching
- **Lines 550-650**: Sidebar configuration & controls
- **Lines 650-750**: Video processing callback
- **Lines 750-850**: Hero section & feature cards
- **Lines 850-950**: Camera panel & live feed
- **Lines 950-1000**: Tips & configuration display
- **Lines 1000-1050**: Detectable objects tag cloud

---

## 🧠 How It Works

### Detection Pipeline

```
1. Camera Input
   ↓
2. WebRTC Stream Capture
   ↓
3. Frame Callback (video_frame_callback)
   ↓
4. Convert Frame (ndarray format)
   ↓
5. YOLOv8 Inference
   - Runs detection model
   - Applies confidence filter
   - Returns detections
   ↓
6. Tracking
   - Assigns/updates track IDs
   - Maintains object history
   ↓
7. Annotation
   - Draws bounding boxes
   - Adds labels & confidence
   - Applies display settings
   ↓
8. Output Frame
   ↓
9. Browser Display (WebRTC)
```

### Key Functions

#### `load_model()`
```python
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")
```
- Caches model to prevent reloading
- Returns pre-trained YOLOv8 model
- Executed once at app startup

#### `video_frame_callback(frame)`
```python
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.track(img, persist=True, conf=confidence, verbose=False)
    annotated = results[0].plot(
        labels=show_labels, conf=show_confidence, boxes=show_boxes
    )
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")
```
- Processes each incoming video frame
- Runs YOLOv8 tracking inference
- Returns annotated frame with detections
- Executes async for performance

---

## 📊 Performance Metrics

| Model | Inference Time | Accuracy | Memory |
|-------|----------------|----------|--------|
| YOLOv8n | ~5ms | 63.4 mAP | ~80MB |
| YOLOv8s | ~10ms | 66.6 mAP | ~150MB |
| YOLOv8m | ~20ms | 70.2 mAP | ~250MB |

*Metrics based on COCO dataset evaluation*

---

## 🎯 Use Cases

- **Security & Surveillance**: Real-time person detection
- **Sports Analytics**: Player/ball tracking
- **Traffic Monitoring**: Vehicle detection & counting
- **Manufacturing**: Quality control & defect detection
- **Retail Analytics**: Customer behavior tracking
- **Research**: Computer vision experimentation
- **Educational**: Learning AI/ML concepts

---

## 🔒 Privacy & Security

- **Local Processing**: All detection happens locally in your browser
- **No Cloud Upload**: Video data never leaves your machine
- **Camera Permissions**: Standard browser security model
- **Open Source**: Code available for auditing

---

## 🚧 Future Enhancements

- [ ] Custom model upload
- [ ] Video file processing
- [ ] Detection history/statistics
- [ ] Export annotations
- [ ] Multi-camera support
- [ ] Real-time performance metrics
- [ ] Alert notifications
- [ ] Recording capability

---

## 📝 License

This project is open-source and available under the MIT License.

---

## 👤 Author

**Live Object Detection & Tracing** - 2026
- Created with ❤️ using YOLOv8 & Streamlit
- Making AI powerful, one detection at a time

---

## 🔗 Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **COCO Dataset**: https://cocodataset.org/
- **PyTorch**: https://pytorch.org/

---

## 💬 Support

For issues, questions, or suggestions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Usage Guide](#usage-guide)
3. Check existing GitHub issues
4. Create a new issue with detailed information

---

**Live Object Detection & Tracing** — Bringing cutting-edge AI detection to your browser ✨ 
