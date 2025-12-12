import cv2
import time
import sys
import os
import numpy as np
import threading
import queue

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import Detector
from tracker import MultiObjectTracker
from fusion import FusionEngine
from monitoring.fps_meter import FPSMeter
from monitoring.logger import get_logger

logger = get_logger("VideoEngine", log_type="json")
COLORS = np.random.uniform(0, 255, size=(100, 3))

class VideoEngine:
    def __init__(self, source, model_path, backend='tensorrt', detection_interval=30):
        self.source = source
        self.detection_interval = detection_interval
        
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.stopped = False

        logger.info(f"Yükleniyor... Backend: {backend}")
        self.detector = Detector(model_path=model_path, backend=backend)
        self.tracker = MultiObjectTracker(tracker_type='KCF')
        self.fusion = FusionEngine(iou_threshold=0.3) # %30 örtüşme
        self.fps_meter = FPSMeter(buffer_len=30)
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Video kaynağı açılamadı: {source}")

    def capture_thread(self):
        logger.info("Capture Thread Başladı")
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            
            if self.input_queue.full():
                try: self.input_queue.get_nowait()
                except queue.Empty: pass
            
            self.input_queue.put(frame)

    def inference_thread(self):
        logger.info("Inference Thread Başladı")
        frame_count = 0
        
        while not self.stopped:
            try:
                frame = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue

            frame_count += 1
            final_boxes = []
            
            # Gerçek anlık durumu hesapla
            current_real_state = "TRACKING"

            if frame_count % self.detection_interval == 0:
                # 1. Detect
                detections, _ = self.detector.detect(frame)
                detections = detections[0] if len(detections) > 0 else []

                # 2. Track Update
                current_tracks = self.tracker.update(frame)

                # 3. Fusion
                fused_boxes, fusion_state = self.fusion.apply_drift_correction(detections, current_tracks)
                current_real_state = fusion_state # FUSED, DRIFT_CORRECT veya INIT
                
                # 4. Re-init
                if len(fused_boxes) > 0:
                    self.tracker.initialize(frame, fused_boxes)
                    final_boxes = fused_boxes
                else:
                    final_boxes = []
            else:
                final_boxes = self.tracker.update(frame)
                current_real_state = "TRACKING"

            if self.output_queue.full():
                try: self.output_queue.get_nowait()
                except queue.Empty: pass
                
            self.output_queue.put((frame, final_boxes, current_real_state))

    def draw_boxes(self, frame, detections, display_state, fps_stats):
        """Kutuları ve bilgileri çizer"""
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            conf = det[4] if len(det) > 4 else 1.0
            cls = int(det[5]) if len(det) > 5 else 0
            
            c = COLORS[cls % len(COLORS)]
            
            # Kutu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            
            # Etiket
            label = f"Obj {cls}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), c, -1)
            cv2.putText(frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        
        # --- DASHBOARD ---
        # Arka plan kutusu
        cv2.rectangle(frame, (10, 10), (250, 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 110), (255, 255, 255), 1)

        # FPS
        cv2.putText(frame, f"FPS: {int(fps_stats['fps'])}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # MODE
        mode_color = (255, 255, 255) # Varsayılan Beyaz
        if "DRIFT" in display_state: mode_color = (0, 0, 255)    # Kırmızı
        elif "FUSED" in display_state: mode_color = (0, 255, 255) # Sarı
        elif "TRACK" in display_state: mode_color = (255, 0, 0)   # Mavi
        
        cv2.putText(frame, f"Mode: {display_state}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # Latency
        cv2.putText(frame, f"P95: {fps_stats['p95']}ms", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame

    def run(self):
        t_cap = threading.Thread(target=self.capture_thread, daemon=True)
        t_inf = threading.Thread(target=self.inference_thread, daemon=True)
        t_cap.start()
        t_inf.start()

        logger.info("Display Loop Başladı")
        
        # --- Sticky State ---
        display_state = "IDLE"
        state_timer = 0
        
        while not self.stopped:
            try:
                data = self.output_queue.get(timeout=1)
                frame, boxes, new_state = data
            except queue.Empty:
                if not t_cap.is_alive(): break
                continue

            self.fps_meter.tick()
            
            # Sticky Logic(Mode'lar gözle görülebilsin diye)
            if "DRIFT" in new_state or "FUSED" in new_state or "INIT" in new_state:
                display_state = new_state
                state_timer = 20 # 20 kare boyunca bu yazıyı tut
            
            # geriye say
            if state_timer > 0:
                state_timer -= 1
            else:
                display_state = "TRACKING" # Varsayılan

            stats = {
                "fps": self.fps_meter.get_fps(),
                "p95": self.fps_meter.get_latency_stats()['p95']
            }

            frame = self.draw_boxes(frame, boxes, display_state, stats)
            cv2.imshow('Edge AI Video Analytics', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
                break

        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Sistem Kapatıldı.")

if __name__ == "__main__":
    video_source = 0
    model_trt = "models/model_int8.engine"
    
    try:
        engine = VideoEngine(source=video_source, model_path=model_trt, backend='tensorrt', detection_interval=30)
        engine.run()
    except Exception as e:
        logger.error(f"Kritik Hata: {e}")