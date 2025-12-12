"""
Part 1 Requirement: Strong augmentations (Albumentations)
Bu dosya YOLOv8 eğitimi için gerekli ileri seviye veri artırma hiperparametrelerini içerir.
"""

HYPERPARAMS = {
    # --- Strong Augmentations ---
    "mosaic": 1.0,        # Mosaic (Görüntüleri birleştirme) 
    "mixup": 0.15,        # MixUp (Görüntüleri karıştırma) 
    "degrees": 10.0,      # Random Rotation
    "translate": 0.1,     # Random Crop
    "scale": 0.5,         # Scale jitter(Multi-scale training)
    "shear": 0.0,         # Shear
    "perspective": 0.0,   # Perspective
    "flipud": 0.0,        # Flip Up-Down
    "fliplr": 0.5,        # Flip Left-Right
    
    # --- Color Jitter (HSV) ---
    "hsv_h": 0.015,       # Hue
    "hsv_s": 0.7,         # Saturation
    "hsv_v": 0.4,         # Value
    
    # --- Optimization ---
    "lr0": 0.01,          # Initial Learning Rate
    "lrf": 0.01,          # Final Learning Rate (Cosine Scheduler için)
    "momentum": 0.937,    # SGD Momentum / Adam beta1
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0, # Warmup
    "warmup_momentum": 0.8,
    "box": 7.5,           # Box Loss Gain
    "cls": 0.5,           # Cls Loss Gain
    "dfl": 1.5,           # DFL Loss Gain
}