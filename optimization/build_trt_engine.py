import tensorrt as trt
import os
import sys


try:
    from calibrate_int8 import YOLOEntropyCalibrator
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from calibrate_int8 import YOLOEntropyCalibrator

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engine_file_path, precision, calib_data_path=None, calib_cache=None):
    print(f"\n TensorRT Engine İnşası Başlıyor: {precision.upper()}")
    print(f" Hedef Dosya: {engine_file_path}")
    
    builder = trt.Builder(TRT_LOGGER)
    # Batch size support
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    #  Workspace size tuning (4GB limit)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30)) 

    # ONNX Parse
    if not os.path.exists(onnx_file_path):
        print(f" Hata: ONNX dosyası bulunamadı -> {onnx_file_path}")
        return

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print(" Hata: ONNX parse edilemedi.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Optimization profiles
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(input_name, (1, 3, 640, 640), (4, 3, 640, 640), (16, 3, 640, 640))
    config.add_optimization_profile(profile)

    # --- PRECISION AYARLARI ---
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(" FP16 modu aktif edildi.")
        else:
            print(" GPU FP16 desteklemiyor, FP32 kullanılacak.")
    
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # FP16 fallback
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            print(" INT8 modu aktif edildi. Kalibrasyon başlıyor...")
            
            # Kalibratörü Bağla
            calibrator = YOLOEntropyCalibrator(
                training_data_path=calib_data_path,
                cache_file=calib_cache,
                batch_size=8 
            )
            config.int8_calibrator = calibrator
        else:
            print(" GPU INT8 desteklemiyor! İşlem atlanıyor.")
            return

    # Build
    print(" Engine derleniyor... (Lütfen bekleyin)")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print(" Engine oluşturulamadı.")
        return

    # Kaydet
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f" Başarılı: {engine_file_path}")

if __name__ == "__main__":
    # Proje Ana Dizini
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ONNX_PATH = os.path.join(ROOT, 'models', 'model.onnx')
    
    # Kalibrasyon Verileri
    CALIB_DATA = os.path.join(ROOT, 'datasets', 'coco128', 'images', 'train2017')
    CALIB_CACHE = os.path.join(ROOT, 'models', 'calibration.cache')
    
    print("="*50)
    print(" OTOMATİK TENSORRT ENGINE ÜRETİMİ (FP16 & INT8)")
    print("="*50)

    # 1. FP16 Üretimi
    FP16_ENGINE = os.path.join(ROOT, 'models', 'model_fp16.engine')
    build_engine(ONNX_PATH, FP16_ENGINE, 'fp16')
    
    print("\n" + "-"*30 + "\n")

    # 2. INT8 Üretimi
    INT8_ENGINE = os.path.join(ROOT, 'models', 'model_int8.engine')
    build_engine(ONNX_PATH, INT8_ENGINE, 'int8', CALIB_DATA, CALIB_CACHE)

    print("\n" + "="*50)
    print(" TÜM İŞLEMLER TAMAMLANDI")
    print(f"1. {FP16_ENGINE}")
    print(f"2. {INT8_ENGINE}")