import cv2
import numpy as np

class MultiObjectTracker:
    def __init__(self, tracker_type='CSRT'):
        """
        tracker_type: 'KCF' (Hizli), 'CSRT' (doğru ama yavaş), 'MOSSE' (Çok hizli)
        """
        self.tracker_type = tracker_type
        self.trackers = [] 
        print(f"📡 Tracker başlatılıyor: {tracker_type}")

    def create_tracker_instance(self):
        # Yöntem 1: Yeni Sürümler (OpenCV 4.5+) -> cv2.legacy altında olabilir(tedbir amaçlı)
        if hasattr(cv2, 'legacy'):
            if self.tracker_type == 'KCF':
                return cv2.legacy.TrackerKCF_create()
            elif self.tracker_type == 'CSRT':
                return cv2.legacy.TrackerCSRT_create()
            elif self.tracker_type == 'MOSSE':
                return cv2.legacy.TrackerMOSSE_create()
        
        # Yöntem 2: Eski Sürümler veya Standart API
        if self.tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif self.tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        
        raise AttributeError(f"OpenCV içinde '{self.tracker_type}' bulunamadı. 'opencv-contrib-python' kurulu mu?")

    def initialize(self, frame, detections):
        """
        Dedektör çalisitğinde tracker'lari sifirlar.
        detections: [[x1, y1, x2, y2, conf, cls], ...]
        """
        self.trackers = []
        
        for det in detections:
            # Tensor -> Numpy
            if hasattr(det, 'cpu'):
                det = det.cpu().numpy()
            
            x1, y1, x2, y2, conf, cls = det
            
            # Koordinatları Integer yap
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # OpenCV Tracker (x, y, w, h) ister
            w = x2 - x1
            h = y2 - y1
            
            # Güvenlik: Boyut kontrolü
            if w <= 0 or h <= 0: 
                continue

            bbox = (x1, y1, w, h)
            
            try:
                tracker = self.create_tracker_instance()
                tracker.init(frame, bbox)
                
                self.trackers.append({
                    'tracker': tracker,
                    'bbox': bbox,
                    'conf': conf,
                    'cls': int(cls)
                })
            except Exception as e:
                print(f"Tracker init hatası: {e}")

    def update(self, frame):
        """
        Dedektörün çalişmadiği karelerde tracker'lari günceller.
        """
        results = []
        active_trackers = []

        for tr_data in self.trackers:
            success, bbox = tr_data['tracker'].update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                # (x, y, w, h) -> (x1, y1, x2, y2)
                results.append([x, y, x + w, y + h, tr_data['conf'], tr_data['cls']])
                
                tr_data['bbox'] = bbox
                active_trackers.append(tr_data)
        
        self.trackers = active_trackers
        
        # Eğer hiç nesne yoksa boş array dön
        if len(results) == 0:
            return np.empty((0, 6))
            
        return np.array(results)