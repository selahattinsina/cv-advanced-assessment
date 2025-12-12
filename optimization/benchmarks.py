import time
import json
import torch
import numpy as np
import pynvml
import pandas as pd
import os
import sys

# Proje ana dizinini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.detector import Detector

class Benchmarker:
    def __init__(self, models_dict, iterations=100, warmup=20):
        self.models = models_dict
        self.iterations = iterations
        self.warmup = warmup
        self.results = {}
        
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.monitor_gpu = True
        except:
            self.monitor_gpu = False
            print("GPU izleme başlatılamadı.")

    def get_gpu_utilization(self):
        if self.monitor_gpu:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return util.gpu, mem.used / 1024**2
        return 0, 0

    def run(self):
        print(f"Benchmark Başlıyor... ({self.iterations} iterasyon)")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

        for name, path in self.models.items():
            print(f"\n--- Testing: {name} ---")
            
            if 'ONNX' in name: backend = 'onnx'
            else: backend = 'tensorrt'

            try:
                detector = Detector(model_path=path, backend=backend)
                
                # Warmup
                print("Warmup...")
                for _ in range(self.warmup):
                    detector.detect(dummy_img)

                # Metrics Containers
                latencies_total = []
                latencies_pre = []
                latencies_infer = []
                latencies_post = []
                gpu_utils = []
                mem_usages = []

                print("Measuring...")
                start_global = time.time()
                
                for _ in range(self.iterations):
                    g_util, g_mem = self.get_gpu_utilization()
                    gpu_utils.append(g_util)
                    mem_usages.append(g_mem)

                    # Inference (Artık detaylı timing dönüyor)
                    _, timings = detector.detect(dummy_img)
                    
                    latencies_total.append(timings['total'])
                    latencies_pre.append(timings['pre_process'])
                    latencies_infer.append(timings['inference'])
                    latencies_post.append(timings['post_process'])

                end_global = time.time()
                fps = self.iterations / (end_global - start_global)

                # Calculate Stats
                def get_stats(arr):
                    return {
                        "avg": round(np.mean(arr), 2),
                        "p50": round(np.percentile(arr, 50), 2),
                        "p95": round(np.percentile(arr, 95), 2)
                    }

                stats_total = get_stats(latencies_total)
                stats_pre = get_stats(latencies_pre)
                stats_post = get_stats(latencies_post)

                # Report Structure (Case uyumlu)
                report = {
                    "Metric": {
                        "Latency (Total)": stats_total,
                        "CPU Latency (Pre-process)": stats_pre,
                        "CPU Latency (Post-process)": stats_post,
                        "Throughput (FPS)": round(fps, 2),
                        "GPU Utilization (%)": round(np.mean(gpu_utils), 2),
                        "GPU Memory (MB)": round(np.mean(mem_usages), 2)
                    }
                }
                
                self.results[name] = report
                print(f"{name}: {stats_total['avg']} ms | {round(fps, 2)} FPS")
                
                del detector
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            except Exception as e:
                print(f"{name} Failed: {e}")

    def save_report(self, output_file="benchmark_results.json"):
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # Basit tablo gösterimi
        print("\n BENCHMARK SUMMARY ")
        print(f"{'Model':<20} | {'FPS':<10} | {'Avg Latency':<15} | {'Pre-Proc':<10} | {'Post-Proc':<10}")
        print("-" * 80)
        for name, data in self.results.items():
            metrics = data["Metric"]
            print(f"{name:<20} | {metrics['Throughput (FPS)']:<10} | {metrics['Latency (Total)']['avg']:<15} | {metrics['CPU Latency (Pre-process)']['avg']:<10} | {metrics['CPU Latency (Post-process)']['avg']:<10}")

if __name__ == "__main__":
    # Üç modeli de test et
    models = {
        'ONNX': 'models/model.onnx',
        'TensorRT (FP16)': 'models/model_fp16.engine',
        'TensorRT (INT8)': 'models/model_int8.engine'
    }
    
    # Dosyaların varlığını kontrol et, yoksa listeden çıkar
    valid_models = {k: v for k, v in models.items() if os.path.exists(v)}
    
    if not valid_models:
        print(" Hiçbir model dosyası bulunamadı! Lütfen önce optimization/build_trt_engine.py çalıştırın.")
    else:
        bencher = Benchmarker(valid_models, iterations=200, warmup=20)
        bencher.run()
        bencher.save_report()