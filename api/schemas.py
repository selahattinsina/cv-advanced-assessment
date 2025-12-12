from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    label: str

class DetectionResponse(BaseModel):
    inference_time_ms: float
    object_count: int
    detections: List[BoundingBox]

class MetricsResponse(BaseModel):
    fps: float
    latency_p95_ms: float
    gpu_utilization_percent: float
    gpu_memory_used_mb: float