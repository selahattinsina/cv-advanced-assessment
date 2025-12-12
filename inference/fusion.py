import numpy as np

class FusionEngine:
    """
    Part 4: Real-Time Video Engine (Detector + Tracker Fusion)
    Görevi: Detection ve Tracking kutularini karsilastirip (IoU), Drift kontrol etmek.
    """
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

    def calculate_iou(self, boxA, boxB):
        # box: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def apply_drift_correction(self, detections, tracks):
        """
        Drift Detection Logic:
        Eğer dedektör yeni bir şey bulduysa ve tracker'in tahmini ile
        örtüşüyorsa (High IoU) -> Tracker'i düzelt (Correction).
        Örtüşmüyorsa (Low IoU) -> Tracker kaymiş demektir, dedektörü kabul et.
        """
        # Eğer hiç tracker yoksa direkt dedektörü kabul et
        if len(tracks) == 0:
            return detections, "INIT"
        
        # Eğer hiç detection yoksa tracker'a güvenmeye devam et (veya sil)
        if len(detections) == 0:
            return tracks, "TRACK"

        final_boxes = []
        matched_indices = set()
        
        # Her bir detection için en uygun tracker'ı bul
        for det in detections:
            best_iou = 0
            best_idx = -1
            
            for i, trk in enumerate(tracks):
                iou = self.calculate_iou(det[:4], trk[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_iou < self.iou_threshold:
                # Drift Detected (veya Yeni Nesne) -> Dedektörü kullan (Re-init)
                final_boxes.append(det)
                state = "DRIFT_CORRECT"
            else:
                # Tracker is good -> Trust Detector for refinement
                # Dedektör koordinatları genelde daha hassastır, tracker'ı güncelleriz.
                final_boxes.append(det) 
                matched_indices.add(best_idx)
                state = "FUSED"
        
        return np.array(final_boxes), state