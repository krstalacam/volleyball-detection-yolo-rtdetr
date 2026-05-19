import os
import cv2
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Proje Kök Dizini ve Yolları
PROJECT_DIR = Path(__file__).resolve().parent.parent
VALID_IMAGES_DIR = PROJECT_DIR / "data" / "valid" / "images"


def get_weights_path(model_dir_name, filename="best.pt"):
    """Trained weights dosyasını arar (Temiz absolute veya YOLO task-prefixed relative)."""
    path_primary = PROJECT_DIR / "runs" / "volleyball_detection" / model_dir_name / "weights" / filename
    if path_primary.exists():
        return path_primary
    path_fallback = PROJECT_DIR / "runs" / "detect" / "runs" / "volleyball_detection" / model_dir_name / "weights" / filename
    if path_fallback.exists():
        return path_fallback
    return None

def run_inference(model_name="yolo11s.pt", n_ims=12, rows=3, conf=0.20, iou=0.45, use_tta=True):
    """
    Belirlenen model ile test/valid veri kümesi üzerinde çıkarım (inference) gerçekleştirir.
    YOLO ve RT-DETR (Ultralytics) mimarilerini destekler.
    """
    print("="*60)
    print(f"🎯 VOLEYBOL TOPU TESPİTİ - {model_name.upper()} İLE TAHMİNLEME 🎯")
    print("="*60)

    # 2. Ultralytics Modelleri (YOLO & RT-DETR) Kontrolü
    stem_name = Path(model_name).stem
    trained_weights_path = get_weights_path(f"{stem_name}_advanced", "best.pt")
    
    if trained_weights_path:
        print(f"[BİLGİ] Eğitilmiş model yüklendi: {trained_weights_path}")
        if "rtdetr" in model_name.lower():
            from ultralytics import RTDETR
            model = RTDETR(trained_weights_path)
        else:
            from ultralytics import YOLO
            model = YOLO(trained_weights_path)
    else:
        print(f"[UYARI] Eğitilmiş model ağırlıkları bulunamadı! Ön eğitimli '{model_name}' yüklenecek.")
        print(f"[BİLGİ] Varsayılan ön eğitimli '{model_name}' ağırlıkları kullanılarak çıkarım yapılacak.")
        if "rtdetr" in model_name.lower():
            from ultralytics import RTDETR
            model = RTDETR(model_name)
        else:
            from ultralytics import YOLO
            model = YOLO(model_name)

    # 3. Çıkarım Yapılacak Görsellerin Seçimi
    if not VALID_IMAGES_DIR.exists():
        print(f"[HATA] Doğrulama klasörü bulunamadı: {VALID_IMAGES_DIR}")
        return
        
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [p for p in VALID_IMAGES_DIR.glob("*") if p.suffix.lower() in valid_extensions]
    
    if len(img_paths) == 0:
        print(f"[HATA] {VALID_IMAGES_DIR} dizininde resim bulunamadı!")
        return
        
    n_ims = min(n_ims, len(img_paths))
    selected_img_paths = random.sample(img_paths, n_ims)
    print(f"[BİLGİ] {len(img_paths)} görsel arasından rastgele {n_ims} tanesi seçildi.")

    # 4. Model Çıkarımı
    print(f"[İPUCU] Çıkarım Ayarları -> Conf: {conf}, IoU: {iou}, TTA (Augment): {use_tta}")
    print("[ÇIKARIM] Model tahminleme yapıyor...")
    
    device = 0 if torch.cuda.is_available() else "cpu"
    
    results = model.predict(
        source=[str(p) for p in selected_img_paths],
        conf=conf,
        iou=iou,
        augment=use_tta,
        device=device,
        verbose=False
    )

    # 5. Profesyonel Izgara Görselleştirme
    cols = int(np.ceil(n_ims / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.flatten() if n_ims > 1 else [axes]
    
    for idx, r in enumerate(results):
        ax = axes[idx]
        annotated_bgr = r.plot(
            conf=True,
            line_width=2,
            font_size=1.0,
            labels=True,
            boxes=True
        )
        
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        n_detected = len(r.boxes)
        
        ax.imshow(annotated_rgb)
        ax.axis("off")
        ax.set_title(f"Görsel #{idx + 1} | {n_detected} Top Tespit Edildi", fontsize=11, fontweight="bold", 
                     color="darkgreen" if n_detected > 0 else "darkred")

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{stem_name.upper()} Voleybol Topu Tespit Sonuçları (Doğrulama Veri Seti)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Gelişmiş Çıkarım ve Tahminleme Scripti (YOLO, RT-DETR)")
    parser.add_argument("--model", type=str, default="yolo11s.pt", 
                        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8s.pt", "rtdetr-l.pt"],
                        help="Tahminleme yapılacak model checkpoint adı.")
    parser.add_argument("--ims", type=int, default=9, help="Görselleştirilecek görsel sayısı.")
    parser.add_argument("--rows", type=int, default=3, help="Izgara satır sayısı.")
    parser.add_argument("--conf", type=float, default=0.20, help="Güven eşiği (Confidence threshold).")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU eşiği.")
    parser.add_argument("--tta", type=bool, default=True, help="Test Time Augmentation (TTA) aktif olsun mu?")
    
    args, unknown = parser.parse_known_args()
    
    run_inference(
        model_name=args.model,
        n_ims=args.ims,
        rows=args.rows,
        conf=args.conf,
        iou=args.iou,
        use_tta=args.tta
    )

if __name__ == "__main__":
    main()

