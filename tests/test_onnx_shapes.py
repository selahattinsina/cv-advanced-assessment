import pytest
import onnxruntime as ort
import numpy as np
import os

MODEL_PATH = "models/model.onnx"

def test_onnx_model_exists():
    assert os.path.exists(MODEL_PATH), "ONNX modeli bulunamadı (önce optimization yapın)"

def test_dynamic_shapes():
    """
    Unit Test: ONNX Dynamic Shapes
    """
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model dosyası yok")

    # CPU Provider ile test (Hızlı ve basit)
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Test 1: Batch Size = 1
    input_b1 = np.zeros((1, 3, 640, 640), dtype=np.float32)
    output_b1 = session.run(None, {input_name: input_b1})
    # Beklenen çıktı: [1, 84, 8400]
    assert output_b1[0].shape == (1, 84, 8400), "Batch 1 shape hatası"

    # Test 2: Batch Size = 2 (Dynamic Axis Testi)
    input_b2 = np.zeros((2, 3, 640, 640), dtype=np.float32)
    try:
        output_b2 = session.run(None, {input_name: input_b2})
        # Beklenen çıktı: [2, 84, 8400]
        assert output_b2[0].shape == (2, 84, 8400), "Batch 2 (Dynamic) shape hatası"
    except Exception as e:
        pytest.fail(f"Dynamic Shape testi başarısız: {e}")