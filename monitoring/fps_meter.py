import time
import collections
import numpy as np

class FPSMeter:
    def __init__(self, buffer_len=30):
        """
        buffer_len: Son kaç kareye bakilarak ortalama alinacak
        """
        self.buffer_len = buffer_len
        self.buffer = collections.deque(maxlen=buffer_len)
        
        # Latency (Gecikme) takibi için son 100 ölçümü tutuyoruz
        self.latencies = collections.deque(maxlen=100) 
        
        self.prev_time = time.time()

    def tick(self):
        curr_time = time.time()
        duration = curr_time - self.prev_time
        self.prev_time = curr_time
        
        fps = 1 / duration if duration > 0 else 0
        self.buffer.append(fps)
        
        # Süreyi ms cinsinden kaydet
        self.latencies.append(duration * 1000) 

    def get_fps(self):
        """Ortalama FPS döner"""
        if not self.buffer:
            return 0.0
        return sum(self.buffer) / len(self.buffer)
    
    def get_latency_stats(self):
        if not self.latencies:
            return {"p50": 0, "p90": 0, "p95": 0}
        
        arr = np.array(self.latencies)
        return {
            "p50": round(np.percentile(arr, 50), 2),
            "p90": round(np.percentile(arr, 90), 2),
            "p95": round(np.percentile(arr, 95), 2)
        }