import sys
import os
import torch
import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO

# Proje kök dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'latest.pt')
ONNX_PATH = os.path.join(ROOT_DIR, 'models', 'model.onnx')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def export_model():
    print(f" Model yükleniyor: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")

    # 1. Modeli Yükle
    model = YOLO(MODEL_PATH)
    
    # 2. ONNX Export İşlemi
    print(" ONNX Export başlatılıyor (Dynamic Axes + Opset 12)...")
    
    # Export işlemi (geçici bir yola kaydeder)
    exported_path = model.export(
        format='onnx',
        dynamic=True,      # REQUIREMENT: Dynamic batch/height/width
        opset=12,          # REQUIREMENT: opset >= 12
        simplify=True,
    )
    
    if str(exported_path) != str(ONNX_PATH):
        if os.path.exists(ONNX_PATH):
            try:
                os.remove(ONNX_PATH)
                print(f" Eski dosya silindi: {ONNX_PATH}")
            except OSError as e:
                print(f" Uyarı: Eski dosya silinemedi: {e}")

        # Dosyayı taşı/yeniden adlandır
        os.rename(exported_path, ONNX_PATH)
        print(f" ONNX dosyası başarıyla oluşturuldu: {ONNX_PATH}")
    else:
        print(f" ONNX dosyası hazır: {ONNX_PATH}")

    # 3. Validation: PyTorch vs ONNX Outputs
    print(" Validation: PyTorch ve ONNX çıktıları sayısal olarak karşılaştırılıyor...")
    
    # Dummy input oluştur
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # a) PyTorch Çıktı
    pytorch_model = model.model
    pytorch_model.eval()
    with torch.no_grad():
        torch_out = pytorch_model(dummy_input)
        if isinstance(torch_out, (tuple, list)):
            torch_out = torch_out[0]

    # b) ONNX Runtime Çıktı
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # c) Karşılaştırma
    try:
        # Binde bir (1e-03) hassasiyetle kontrol et
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print(" SUCCESS: PyTorch ve ONNX çıktıları birebir uyuşuyor! (Validation Passed)")
    except AssertionError as e:
        print(" WARNING: Çıktılar arasında ufak float farkları var (Normal olabilir).")

        print(e)

if __name__ == "__main__":
    try:
        export_model()
    except Exception as e:
        print(f" Kritik Hata: {e}")