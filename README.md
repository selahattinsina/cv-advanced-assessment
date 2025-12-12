# CV Advanced Assessment - Edge AI Object Detection System

A comprehensive computer vision project implementing YOLOv8-based object detection with TensorRT optimization, real-time video processing, and RESTful API services.

## Project Overview

This project provides an end-to-end solution for object detection and tracking, featuring:
- **Model Training**: YOLOv8 training with advanced augmentations
- **Model Optimization**: ONNX export and TensorRT engine conversion (FP16/INT8)
- **Real-time Inference**: Video processing with detection, tracking, and fusion
- **RESTful API**: FastAPI server for object detection endpoints
- **Monitoring**: FPS tracking, GPU metrics, and JSON logging

## Project Structure

```
cv-advanced-assessment/
├── api/                    # FastAPI server and schemas
│   ├── server.py          # Main API server with /detect, /metrics, /health
│   ├── schemas.py         # Pydantic models for API responses
│   └── docker/            # Docker configuration
├── datasets/              # Training datasets
│   └── coco128/          # COCO128 dataset (80 classes)
├── inference/             # Inference pipeline components
│   ├── detector.py       # YOLOv8 detector wrapper
│   ├── tracker.py        # Multi-object tracker (KCF)
│   ├── fusion.py         # Detection-Tracking fusion engine
│   ├── video_engine.py   # Real-time video processing engine
│   └── utils.py          # Utility functions
├── models/                # Trained models and optimized engines
│   ├── latest.pt         # Best trained model weights
│   ├── model.onnx        # ONNX exported model
│   ├── model_fp16.engine # TensorRT FP16 engine
│   └── model_int8.engine # TensorRT INT8 engine
├── monitoring/            # Monitoring and logging
│   ├── logger.py         # JSON logger implementation
│   └── fps_meter.py      # FPS and latency tracking
├── optimization/          # Model optimization scripts
│   ├── export_to_onnx.py # ONNX export with validation
│   ├── build_trt_engine.py # TensorRT engine builder
│   ├── calibrate_int8.py # INT8 calibration
│   └── benchmarks.py     # Performance benchmarking
├── training/              # Model training
│   ├── train.py          # Main training script
│   ├── augmentations.py  # Advanced augmentation hyperparameters
│   ├── dataset.yaml      # Dataset configuration
│   └── logs/             # Training logs and visualizations
└── tests/                # Unit tests
    ├── test_inference.py
    ├── test_onnx_shapes.py
    └── test_tracker.py
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for TensorRT)
- NVIDIA drivers and CUDA toolkit
- TensorRT library

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/selahattinsina/cv-advanced-assessment
   cd cv-advanced-assessment
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained YOLOv8 weights** (if not present)
   - The project expects `yolov8n.pt` in the root directory
   - Ultralytics will auto-download if missing

## Usage Guide

Follow these steps in order:

### Part 1: Model Training

Train a YOLOv8 model with advanced augmentations:

```bash
python training/train.py
```

**Requirements:**
- Strong augmentations (Mosaic, MixUp, CutOut, MotionBlur, RandomCrop, ColorJitter)
- Multi-scale training
- EMA (Exponential Moving Average)
- Cosine LR schedule
- AMP (mixed precision)

**Outputs:**
- `models/latest.pt` - Best model weights
- `training/logs/` - Training logs including:
  - Loss curves
  - mAP@0.5 and mAP@0.5:0.95
  - Confusion Matrix
  - TensorBoard logs

### Part 2: Model Optimization Pipeline

#### Step 1: PyTorch → ONNX

```bash
python optimization/export_to_onnx.py
```

**Features:**
- Dynamic axes (batch size, image height/width)
- Opset ≥ 12
- Validates ONNX outputs match PyTorch

**Output:** `models/model.onnx`

#### Step 2: ONNX → TensorRT Engines

```bash
python optimization/build_trt_engine.py
```

**Features:**
- Generates both FP16 and INT8 TensorRT engines
- Optimization profiles (min/opt/max resolution)
- Workspace size tuning
- Batch size support
- INT8 calibration (entropy-based, 200-500 images)
- Saves calibration cache → `models/calibration.cache`

**Outputs:**
- `models/model_fp16.engine`
- `models/model_int8.engine`
- `models/calibration.cache`

#### Step 3: Benchmarking

```bash
python optimization/benchmarks.py
```

**Metrics:**
- Latency (avg / p50 / p95)
- Throughput (FPS)
- GPU Utilization (pynvml)
- CPU Latency (pre/post-processing)
- Warmup (≥ 10 iterations)

**Output:** `benchmark_results.json`

### Part 3: Unit Tests

```bash
pytest -q
```

**Tests:**
- ONNX dynamic shapes
- TensorRT engine loading
- Warm-up
- Tracker drift
- Pre/Post-process consistency
- I/O shape validation

### Part 4: Multi-Backend Inference Engine

The `Detector` class (used by video engine and API) supports:
- PyTorch inference
- ONNX Runtime (CPU / CUDA / TensorRT EP)
- Native TensorRT (Python API)

**Features:**
- Consistent pre/post-processing
- Custom NMS
- Batch inference
- Warm-up
- Timing statistics

### Part 5: Real-Time Video Engine

```bash
python inference/video_engine.py
```

**Features:**
- Hybrid tracking (KCF tracker)
- Drift detection (IoU-based, reinitialize if IoU < 0.5)
- Multi-threaded architecture (Capture → Inference → Tracking → Display)
- Detection-Tracking fusion

### Part 6: API Deployment

```bash
python api/server.py
```

**Endpoints:**
- `GET /health` - Health check
- `POST /detect` - Object detection (returns bbox + confidence + inference time)
- `GET /metrics` - System metrics (latency, FPS, GPU usage)

**Features:**
- Auto-loads TensorRT engine on startup
- GPU-enabled
- Docker support (NVIDIA TensorRT Runtime image)

### Part 7: Performance Monitoring

Implemented in `monitoring/`:
- FPS meter
- GPU memory + utilization logger
- Latency histogram (p50/p90/p95)
- JSON logging
- Moving average latency (window=100)

## Requirements

See `requirements.txt` for full dependency list:

**API & Server:**
- `fastapi`, `uvicorn`, `python-multipart`, `pydantic`

**AI Core & Processing:**
- `ultralytics`, `opencv-python-headless`, `numpy<2.0`, `pynvml`, `pycuda`

**ONNX Support:**
- `onnx`, `onnxruntime-gpu`

**Utilities & Logging:**
- `colorama`, `psutil`, `tqdm`

**Testing:**
- `pytest`


