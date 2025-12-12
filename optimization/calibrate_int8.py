import tensorrt as trt
import os
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import glob

class YOLOEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data_path, cache_file, batch_size=1, height=640, width=640):
        # Entropy-based calibration
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.cache_file = cache_file
        self.shape = (batch_size, 3, height, width)
        self.batch_size = batch_size
        self.current_index = 0
        
        # Resim yollarını topla
        extensions = ['*.jpg', '*.jpeg', '*.png']
        self.img_paths = []
        for ext in extensions:
            self.img_paths.extend(glob.glob(os.path.join(training_data_path, ext)))
        
        # Load calibration dataset (200-500 images)
        # Elimizde COCO128 olduğu için 128 kullanıyorum
        self.img_paths = self.img_paths[:500]
        
        print(f" Kalibrasyon için {len(self.img_paths)} resim bulundu. (Hedef: 200-500)")
        
        # GPU belleği ayır (Input boyutu kadar)
        self.device_input = cuda.mem_alloc(batch_size * 3 * height * width * 4) # 4 bytes per float32

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """TensorRT bu fonksiyonu döngü içinde çağırır"""
        if self.current_index + self.batch_size > len(self.img_paths):
            return None

        batch_imgs = []
        for i in range(self.batch_size):
            img_path = self.img_paths[self.current_index + i]
            
            # --- Preprocessing ---
            # Resize + Normalize
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Resize
            img = cv2.resize(img, (640, 640))
            
            # BGR -> RGB & CHW Conversion
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            
            # Normalize 0-1
            img = img.astype(np.float32) / 255.0
            batch_imgs.append(img)
        
        self.current_index += self.batch_size
        
        # Veriyi tek bir blok haline getir
        batch_data = np.ascontiguousarray(np.array(batch_imgs)).ravel()
        
        # CPU -> GPU Kopyala
        cuda.memcpy_htod(self.device_input, batch_data)
        
        return [int(self.device_input)]

    def read_calibration_cache(self):
        """Eğer cache dosyasi varsa onu kullan"""
        if os.path.exists(self.cache_file):
            print(f" Kalibrasyon cache dosyası okunuyor: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Case İsteri: Save calibration cache"""
        print(f" Kalibrasyon cache dosyası yazılıyor: {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)