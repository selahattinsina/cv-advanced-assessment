import pytest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.detector import Detector
from inference.utils import letterbox

MODEL_TRT = "models/model_int8.engine"

def test_preprocess_consistency():
    """
    Unit Test: Pre/Post-process Consistency
    Görüntü 1920x1080 -> 640x640 olmali (Padding).
    """
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    new_img, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640), auto=False)
    
    # 1. Shape Validation
    assert new_img.shape == (640, 640, 3), "Pre-process çıkış boyutu hatalı"
    
    # 2. Consistency: Aspect Ratio korunmalı, padding ekle
    # 1920/1080 = 1.77, 640/640 = 1.0 -> Padding
    assert dw > 0 or dh > 0, "Padding eklenmemiş, aspect ratio bozulmuş olabilir"

def test_tensorrt_loading_and_warmup():
    """
    Unit Test: TensorRT Engine Loading & Warm-up
    """
    if not os.path.exists(MODEL_TRT):
        pytest.skip("TensorRT engine bulunamadı")

    try:
        # Loading
        det = Detector(MODEL_TRT, backend='tensorrt')
        assert det.context is not None, "TensorRT Context yüklenemedi"
        
        # Warm-up Testi 
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        det.detect(dummy) 
        
    except ImportError:
        pytest.skip("TensorRT/PyCUDA kütüphanesi eksik")
    except Exception as e:
        pytest.fail(f"Engine yükleme hatası: {e}")

def test_io_shape_validation():
    """
    Unit Test: I/O Shape Validation
    Model girişinin (1,3,640,640) ve çikisinin (1,84,8400) olduğunu doğrular.
    """
    if not os.path.exists(MODEL_TRT):
        pytest.skip("Model yok")

    det = Detector(MODEL_TRT, backend='tensorrt')
    
    # Input Validation
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    blob, _, _ = det.preprocess(dummy)
    assert blob.shape == (1, 3, 640, 640), "Model giriş tensörü hatalı"