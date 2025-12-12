from ultralytics import YOLO
import os
import shutil
import sys

# Augmentations parametrelerini çek
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from augmentations import HYPERPARAMS

def train_model():
    print(" Part 1: Model Training Başlıyor...")
    
    # Proje kök dizini
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 1. Modeli Başlat (YOLOv8n - Nano model hızlı eğitim için)
    model = YOLO('yolov8n.pt') 

    # 2. Eğitimi Başlat
    # Case Requirements:
    # - Cosine LR schedule (cos_lr=True)
    # - AMP mixed precision (amp=True - Default)
    # - Multi-scale training (imgsz değişkenliği ve rect=False ile sağlanır, scale aug aktif)
    # - Output: Tensorboard logs (project/name args ile)
    
    results = model.train(
        data=os.path.join(ROOT_DIR, 'training', 'dataset.yaml'),
        epochs=10,             # Hızlı sonuç için 10 (Normalde 100+)
        imgsz=640,
        batch=16,
        project=os.path.join(ROOT_DIR, 'runs/detect'), # Geçici kayıt yeri
        name='train_run',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=42,
        cos_lr=True,           # Requirement: Cosine LR schedule
        amp=True,              # Requirement: AMP (Mixed Precision)
        **HYPERPARAMS          # Requirement: Strong Augmentations (Mosaic, MixUp, etc.)
    )

    print(" Eğitim Tamamlandı. Dosyalar düzenleniyor...")

    # --- POST-TRAINING FILE MANAGEMENT ---
    
    # Hedef Klasörler
    MODELS_DIR = os.path.join(ROOT_DIR, 'models')
    LOGS_DIR = os.path.join(ROOT_DIR, 'training', 'logs')
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Kaynak Dosyalar
    RUN_DIR = os.path.join(ROOT_DIR, 'runs', 'detect', 'train_run')
    BEST_MODEL = os.path.join(RUN_DIR, 'weights', 'best.pt')
    
    # 1. Output: models/latest.pt
    TARGET_MODEL = os.path.join(MODELS_DIR, 'latest.pt')
    if os.path.exists(BEST_MODEL):
        shutil.copy(BEST_MODEL, TARGET_MODEL)
        print(f" Model taşındı: {TARGET_MODEL}")
    else:
        print(" Hata: best.pt bulunamadı!")

    # 2. Output: Training logs /training/logs/
    # TensorBoard event dosyalarını ve CSV loglarını taşı
    for filename in os.listdir(RUN_DIR):
        if filename.startswith("events.out") or filename.endswith(".csv") or filename == "args.yaml":
            src = os.path.join(RUN_DIR, filename)
            dst = os.path.join(LOGS_DIR, filename)
            shutil.copy(src, dst)
    
    # Confusion Matrix ve Loss Eğrileri
    for plot_file in ["confusion_matrix.png", "results.png"]:
        src = os.path.join(RUN_DIR, plot_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(LOGS_DIR, plot_file))

    print(f" Loglar taşındı: {LOGS_DIR}")
    print(" Part 1 Başarıyla Tamamlandı!")

if __name__ == "__main__":
    train_model()