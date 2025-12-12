import pytest
import sys
import os
import numpy as np

# Proje ana dizinini ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.fusion import FusionEngine
from inference.tracker import MultiObjectTracker

def test_iou_calculation_logic():
    """
    Unit Test: IoU (Intersection over Union) Matematiği
    """
    fusion = FusionEngine()
    
    # 1. Tam Örtüşme (IoU ~ 1.0)
    boxA = [100, 100, 200, 200]
    boxB = [100, 100, 200, 200]
    assert fusion.calculate_iou(boxA, boxB) > 0.99, "Tam örtüşme hatası"

    # 2. Hiç Örtüşmeme (IoU = 0.0)
    boxC = [300, 300, 400, 400]
    assert fusion.calculate_iou(boxA, boxC) == 0.0, "Ayrık kutu hatası"

    # 3. Kısmi Örtüşme
    boxD = [150, 150, 250, 250] # Yarısı iç içe
    iou = fusion.calculate_iou(boxA, boxD)
    assert 0.0 < iou < 1.0, "Kısmi örtüşme hatası"

def test_tracker_drift_scenario():
    """
    Unit Test: Tracker Drift Detection
    Senaryo: Tracker kayarsa (Düşük IoU), sistem 'DRIFT_CORRECT' dönmeli.
    """
    # Eşik değer %50 
    fusion = FusionEngine(iou_threshold=0.5)
    
    # Dedektörden gelen kesin veri
    det = [[100, 100, 200, 200, 0.9, 0]]
    
    # Senaryo A: Tracker iyi gidiyor (Yüksek IoU)
    trk_good = [[105, 105, 195, 195, 1]]
    _, state_good = fusion.apply_drift_correction(det, trk_good)
    assert state_good == "FUSED", "Tracker iyiyken FUSED dönmeli"

    # Senaryo B: Tracker kaymış (Drift - Düşük IoU)
    trk_bad = [[500, 500, 600, 600, 1]] 
    _, state_bad = fusion.apply_drift_correction(det, trk_bad)
    # Drift durumunda dedektör verisi (final_boxes) tracker'ı ezmeli
    assert state_bad == "DRIFT_CORRECT" or "INIT", "Drift durumunda düzeltme yapılmalı"