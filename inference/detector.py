import cv2
import torch
import numpy as np
import time
import os
import onnxruntime as ort
from ultralytics import YOLO

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None

try:
    from inference.utils import letterbox, non_max_suppression
except ImportError:
    from utils import letterbox, non_max_suppression

class Detector:
    def __init__(self, model_path, backend='tensorrt', device='cuda'):
        self.backend = backend
        self.device = device
        self.model_path = model_path
        
        print(f"Detector başlatılıyor... Backend: {backend.upper()} | Model: {model_path}")

        # --- PYTORCH BACKEND ---
        if backend == 'pytorch':
            try:
                self.model = YOLO(model_path)
                # Raw model erişimi (Custom NMS için)
                self.pytorch_model = self.model.model
                self.pytorch_model.eval()
                if device == 'cuda' and torch.cuda.is_available():
                    self.pytorch_model.to('cuda')
                else:
                    self.pytorch_model.to('cpu')
            except Exception as e:
                raise RuntimeError(f"PyTorch modeli yüklenemedi: {e}")

        # --- ONNX BACKEND ---
        elif backend == 'onnx':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            try:
                self.session = ort.InferenceSession(model_path, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
            except Exception as e:
                raise RuntimeError(f"ONNX modeli yüklenemedi: {e}")
        
        # --- TENSORRT BACKEND ---
        elif backend == 'tensorrt':
            if trt is None:
                raise ImportError("TensorRT kütüphanesi yüklü değil!")
                
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            
            try:
                with open(model_path, "rb") as f:
                    self.engine = self.runtime.deserialize_cuda_engine(f.read())
            except Exception as e:
                raise RuntimeError(f"Engine dosyası okunamadı: {e}")
            
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
            
        else:
            raise ValueError(f"Desteklenmeyen backend: {backend}")

        # Warmup
        print("Warmup yapılıyor...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self.detect(dummy_img)
            print("Detector hazır!")
        except Exception as e:
            print(f"Warmup uyarısı: {e}")

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        num_io = engine.num_io_tensors
        
        for i in range(num_io):
            tensor_name = engine.get_tensor_name(i)
            
            # Static shape varsayımı 
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                # Dynamic shape için boyut bildirme
                self.context.set_input_shape(tensor_name, (1, 3, 640, 640))
                size = 1 * 3 * 640 * 640
                dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            else:
                size = 1 * 84 * 8400
                dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            # TensorRT 10.x+ Adres Ataması
            self.context.set_tensor_address(tensor_name, int(device_mem))

            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'name': tensor_name, 'host': host_mem, 'device': device_mem})
                
        return inputs, outputs, bindings, stream

    def preprocess(self, img):
        # Ortak Pre-processing
        image, ratio, dwdh = letterbox(img, new_shape=(640, 640), auto=False)
        image = image.transpose((2, 0, 1))[::-1]  
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # PyTorch için Tensor dönüşümü
        if self.backend == 'pytorch':
            tensor = torch.from_numpy(image)
            if self.device == 'cuda':
                tensor = tensor.to('cuda')
            return tensor, ratio, dwdh
            
        return image, ratio, dwdh

    def detect(self, img):
        # 1. Pre-process
        t0 = time.time()
        blob, ratio, dwdh = self.preprocess(img)
        t1 = time.time()
        
        # 2. Inference
        if self.backend == 'pytorch':
            # Raw output alıyoruz (Consistent Post-processing için)
            with torch.no_grad():
                preds = self.pytorch_model(blob)
                # Output bazen tuple/list  (YOLO architecture'a göre)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                    
        elif self.backend == 'onnx':
            preds = self.session.run(None, {self.input_name: blob})[0]
            
        elif self.backend == 'tensorrt':
            np.copyto(self.inputs[0]['host'], blob.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            if hasattr(self.context, 'execute_async_v3'):
                self.context.execute_async_v3(stream_handle=self.stream.handle)
            else:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
            preds = self.outputs[0]['host'].reshape(1, 84, 8400)
        
        t2 = time.time()

        # 3. Post-process (NMS fonksiyonu)
        preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)
        t3 = time.time()
        
        timings = {
            'pre_process': (t1 - t0) * 1000,
            'inference': (t2 - t1) * 1000,
            'post_process': (t3 - t2) * 1000,
            'total': (t3 - t0) * 1000
        }
        
        return preds, timings

    def __call__(self, img):
        """
        API Uyumluluğu için: detector(frame) şeklinde çağrilir
        """
        preds, _ = self.detect(img)
        return preds

if __name__ == "__main__":
    # TEST ALANI
    img_path = "datasets/coco128/images/train2017/000000000009.jpg"
    img = cv2.imread(img_path)
    if img is None: img = np.zeros((640,640,3), dtype=np.uint8)

    print("\n--- Test 1: TensorRT (INT8) ---")
    try:
        det = Detector("models/model_int8.engine", backend='tensorrt')
        res, t = det.detect(img)
        print(f"TRT Time: {t['total']:.2f} ms")
    except Exception as e:
        print(f"Hata: {e}")

    print("\n--- Test 2: PyTorch (Validation) ---")
    try:
        det = Detector("models/latest.pt", backend='pytorch')
        res, t = det.detect(img)
        print(f"PyTorch Time: {t['total']:.2f} ms")
        print("API Test (Call):", len(det(img)[0]), "nesne bulundu")
    except Exception as e:
        print(f"Hata: {e}")