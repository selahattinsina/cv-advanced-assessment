from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import cv2
import numpy as np
import sys
import os
import pynvml
import time

# Proje ana dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.detector import Detector

from monitoring.logger import get_logger

try:
    from api.schemas import DetectionResponse, MetricsResponse
except ImportError:
    from schemas import DetectionResponse, MetricsResponse

# --- LOGGER ---
# Part 6: JSON Logging
logger = get_logger("API", log_type="json")

app = FastAPI(title="Edge AI Detection API", description="YOLOv8 + TensorRT Inference Server")

# --- AYARLAR ---
MODEL_PATH = "models/model_int8.engine" 
BACKEND = "tensorrt"

# Global Değişkenler
detector = None
gpu_handle = None
last_inference_time = 0.0

@app.on_event("startup")
async def startup_event():
    """Requirement: Auto-load TensorRT engine"""
    global detector, gpu_handle
    
    # 1. GPU Monitor Başlat
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        logger.info("GPU İzleme Başlatıldı")
    except Exception as e:
        logger.warning(f"GPU İzleme başlatılamadı: {e}")

    # 2. Modeli Yükle
    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Model yükleniyor: {MODEL_PATH}...")
            detector = Detector(model_path=MODEL_PATH, backend=BACKEND)
            logger.info("Model ve TensorRT Engine başarıyla yüklendi!")
        except Exception as e:
            logger.critical(f"Kritik Hata: Model yüklenemedi -> {e}")
    else:
        logger.error(f"Hata: Model dosyası bulunamadı -> {MODEL_PATH}")

@app.get("/health")
def health_check():
    """Requirement: /health endpoint"""
    if detector is not None:
        return {"status": "healthy", "backend": BACKEND, "model": MODEL_PATH}
    else:
        logger.error("Health check failed: Model not loaded")
        raise HTTPException(status_code=503, detail="System unhealthy: Model not loaded")

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    """Requirement: /detect -> bbox + confidence + inference time"""
    global last_inference_time
    
    if not detector:
        logger.error("Detect request failed: Service unavailable")
        raise HTTPException(status_code=503, detail="Model servisi aktif değil.")

    # Dosyayı oku
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Image decode failed: {e}")
        raise HTTPException(status_code=400, detail="Geçersiz resim dosyası.")

    # Inference
    try:
        detections, timings = detector.detect(img)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Inference hatası")
    
    # Sonuçları formatla
    results = []
    if len(detections) > 0:
        for det in detections[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            results.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(conf, 2),
                "class_id": int(cls),
                "label": f"Class {int(cls)}"
            })

    # Metrikler için son süreyi kaydet
    last_inference_time = timings['total']

    # Her başarılı isteği logla
    logger.info(f"Detection Success: {len(results)} objects in {round(timings['total'], 2)}ms")

    return {
        "inference_time_ms": round(timings['total'], 2),
        "object_count": len(results),
        "detections": results
    }

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Requirement: /metrics (latency, FPS, GPU usage)"""
    
    # GPU Kullanımı
    gpu_util = 0.0
    mem_used = 0.0
    if gpu_handle:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_util = float(util.gpu)
            mem_used = float(mem.used / 1024**2)
        except Exception as e:
            logger.warning(f"Metrics read failed: {e}")
    
    # FPS Hesabı
    current_fps = 0.0
    if last_inference_time > 0:
        current_fps = 1000.0 / last_inference_time

    return {
        "fps": round(current_fps, 2),
        "latency_p95_ms": round(last_inference_time, 2),
        "gpu_utilization_percent": gpu_util,
        "gpu_memory_used_mb": round(mem_used, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)