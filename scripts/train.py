import os
import torch
import shutil
import argparse
from pathlib import Path

# Proje Kök Dizini ve Dataset Yolu (Windows ve Linux uyumlu)
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML_PATH = PROJECT_DIR / "data" / "data.yaml"

def run_training(model_name="yolo11s.pt", epochs=100, batch=16, patience=15, imgsz=640, progress_callback=None):
    """
    Belirli bir YOLO veya RT-DETR modelini gelişmiş hiperparametreler ile eğitir.
    
    Args:
        model_name (str): Eğitilecek model adı (yolo11s.pt, yolov8s.pt, rtdetr-l.pt vb.)
        epochs (int): Maksimum epoch sayısı.
        batch (int): Batch boyutu.
        patience (int): Erken durdurma sabrı.
        imgsz (int): Eğitim çözünürlüğü.
    """
    print("="*60)
    print(f"VOLEYBOL TOPU TESPITI - {model_name.upper()} EGITIMI BASLIYOR")
    print("="*60)

    # 1. GPU ve Cihaz Kontrolü
    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
        print(f"[CIHAZ] GPU AKTİF: {device_name} (device={device})")
    else:
        device = "cpu"
        print("[CIHAZ] GPU bulunamadı! İşlemler CPU üzerinde devam edecek (Eğitim yavaş olabilir).")
    
    # 2. Dataset Kontrolü
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"[HATA] 'data.yaml' dosyası {DATA_YAML_PATH} yolunda bulunamadı! Lütfen veri setini hazırlayın.")
    print(f"[BİLGİ] Veri seti tanım dosyası: {DATA_YAML_PATH}")

    # 3. Model Yükleme (YOLO vs RT-DETR Seçimi)
    stem_name = Path(model_name).stem
    run_name = f"{stem_name}_advanced"
    
    print(f"[BİLGİ] '{model_name}' ön eğitimli ağırlıkları indiriliyor/yükleniyor...")
    if "rtdetr" in model_name.lower():
        from ultralytics import RTDETR
        model = RTDETR(model_name)
    else:
        from ultralytics import YOLO
        model = YOLO(model_name)

    if progress_callback is not None:
        progress_state = {"epoch": 0, "batch": 0}

        def _emit_progress(trainer):
            try:
                train_loader = getattr(trainer, "train_loader", None)
                total_batches = max(1, len(train_loader)) if train_loader is not None else 1
                progress_epoch = progress_state["epoch"] + (progress_state["batch"] / total_batches)
                progress_callback(min(float(progress_epoch), float(epochs)), int(epochs))
            except Exception:
                pass

        def _on_train_start(trainer):
            progress_callback(0.05, int(epochs))

        def _on_epoch_start(trainer):
            progress_state["epoch"] = int(getattr(trainer, "epoch", progress_state["epoch"]))
            progress_state["batch"] = 0
            progress_callback(float(progress_state["epoch"]), int(epochs))

        def _on_batch_end(trainer):
            progress_state["batch"] += 1
            _emit_progress(trainer)

        def _on_epoch_end(trainer):
            progress_state["epoch"] = int(getattr(trainer, "epoch", progress_state["epoch"])) + 1
            progress_state["batch"] = 0
            progress_callback(min(float(progress_state["epoch"]), float(epochs)), int(epochs))

        model.add_callback("on_train_start", _on_train_start)
        model.add_callback("on_train_epoch_start", _on_epoch_start)
        model.add_callback("on_train_batch_end", _on_batch_end)
        model.add_callback("on_train_epoch_end", _on_epoch_end)

    # 4. Gelişmiş Eğitim Hiperparametreleri
    # Voleybol gibi küçük ve hızlı hareket eden nesnelerin tespiti için özel olarak optimize edilmiştir.
    
    is_transformer = "rtdetr" in model_name.lower()
    
    training_args = {
        "data": str(DATA_YAML_PATH),       # data.yaml dosya yolu
        "epochs": epochs,                  # Belirlenen epoch sayısı
        "patience": patience,              # Erken durdurma sabrı
        "imgsz": imgsz,                    # Görsel çözünürlüğü
        "batch": batch,                    # GPU belleğine göre batch boyutu
        "device": device,                  # Otomatik seçilen donanım (Cuda veya CPU)
        
        # Optimizasyon ve Öğrenme Oranı Ayarları
        "optimizer": "AdamW",              # Küçük veri setleri için kararlı yakınsar
        "lr0": 0.0001 if is_transformer else 0.001,  # Transformer için çok daha düşük LR (gradyan çöküşünü önler)
        "cos_lr": True,                    # Cosine annealing öğrenme oranı zamanlayıcısı ile kararlı iniş
        "warmup_epochs": 5 if is_transformer else 3, # Transformer için daha uzun ısınma süresi
        
        # Veri Artırımı (Augmentation) Ayarları (Aşırı öğrenmeyi -overfitting- engeller)
        # NOT: Mozaik artırımı CNN'ler için harikayken, küçük veri setlerindeki Transformer'ların dikkat haritasını bozar.
        "mosaic": 0.0 if is_transformer else 1.0,   # Transformer için mozaik kapalı
        "mixup": 0.0 if is_transformer else 0.1,    # Transformer için mixup kapalı
        "degrees": 15.0,                   # Görüntüyü döndürme
        "translate": 0.15,                 # Kaydırma
        "scale": 0.5,                      # Ölçekleme
        "shear": 2.0,                      # Yamultma
        "flipud": 0.2,                     # Dikey çevirme
        "fliplr": 0.5,                     # Yatay çevirme
        "erasing": 0.4,                    # Rastgele piksel silme
        
        # İnce Ayar
        "close_mosaic": 0 if is_transformer else 10, # Mozaik kapalıysa bu ayarı da sıfırlıyoruz
        
        # Proje Kayıt Yapısı
        "project": str(PROJECT_DIR / "runs" / "volleyball_detection"), # Kayıt ana dizini (Absolute path ile YOLO task öneki eklemez)
        "name": run_name,                  # Model bazlı çalışma ismi (Örn: yolo11s_advanced, rtdetr-l_advanced)
        "exist_ok": True,                  # Aynı isimli klasör varsa üzerine yazar/devam eder
        "plots": True                      # Eğitim metrik grafiklerini kaydeder
    }

    # RT-DETR mozaik kapatmayı doğrudan desteklemediği için gerekirse filtreleyebiliriz, 
    # ancak ultralytics bunu otomatik yönetir.
    
    print("\n" + "="*50)
    print("EGITIM HIPERPARAMETRELERI:")
    for k, v in training_args.items():
        print(f" - {k}: {v}")
    print("="*50 + "\n")

    # EĞİTİMİ BAŞLAT
    try:
        results = model.train(**training_args)
        print("\n" + "="*60)
        print("EGITIM BASARIYLA TAMAMLANDI!")
        print(f"[BASARI] En iyi model agirligi kaydedildi: runs/volleyball_detection/{run_name}/weights/best.pt")
        print(f"[BASARI] Son model agirligi kaydedildi: runs/volleyball_detection/{run_name}/weights/last.pt")
        print("="*60)
        
    except Exception as e:
        print(f"\n[EĞİTİM HATASI] Eğitim esnasında bir hata oluştu: {e}")
        print("[ÖNERİ] Eğer GPU Bellek Yetmezliği (CUDA Out Of Memory) alıyorsanız, batch size değerini 8 veya 4 yapın.")

def main():
    # Argümanları parse et
    parser = argparse.ArgumentParser(description="Gelişmiş Model Eğitim Scripti (YOLO & RT-DETR)")
    parser.add_argument("--model", type=str, default="yolo11s.pt", 
                        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8s.pt", "rtdetr-l.pt"],
                        help="Eğitilecek model adı (YOLO veya RT-DETR).")
    parser.add_argument("--epochs", type=int, default=100, help="Maksimum epoch sayısı.")
    parser.add_argument("--batch", type=int, default=16, help="Batch boyutu.")
    parser.add_argument("--patience", type=int, default=15, help="Erken durdurma sabrı.")
    parser.add_argument("--imgsz", type=int, default=640, help="Eğitim görsel boyutu.")
    
    args, unknown = parser.parse_known_args()
    
    run_training(
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        imgsz=args.imgsz
    )

if __name__ == "__main__":
    main()

