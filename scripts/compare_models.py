import os
import sys
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Windows terminallerinde Türkçe karakter ve emoji basarken oluşan kodlama hatasını önlemek için
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Proje Kök Dizini ve Yolları
PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "runs" / "volleyball_detection"

def compare_trained_models():
    """
    Eğitilmiş modellerin (YOLOv8, YOLO11, RT-DETR) sonuçlarını 
    runs/volleyball_detection altındaki results.csv dosyalarından okur,
    akademik karşılaştırma grafikleri çizer ve performans tablosu hazırlar.
    """
    print("="*60)
    print("📈 ÇOKLU MODEL KARŞILAŞTIRMA VE AKADEMİK DEĞERLENDİRME 📈")
    print("="*60)

    # 1. Her iki olası dizinde de results.csv dosyalarını ara ve birleştir
    csv_paths = []
    
    # Temiz absolute dizin
    if RUNS_DIR.exists():
        csv_paths.extend(list(RUNS_DIR.glob("**/results.csv")))
        
    # Windows fallback göreli YOLO dizini
    fallback_runs_dir = PROJECT_DIR / "runs" / "detect" / "runs" / "volleyball_detection"
    if fallback_runs_dir.exists():
        csv_paths.extend(list(fallback_runs_dir.glob("**/results.csv")))
        
    # Tekrarlanan yolları kaldır (set ile)
    csv_paths = list(set(csv_paths))
    
    if len(csv_paths) == 0:
        print("[UYARI] Eğitilmiş herhangi bir modele ait 'results.csv' dosyası bulunamadı!")
        print_training_instructions()
        return

    print(f"[BİLGİ] {len(csv_paths)} farklı model eğitimi sonucu tespit edildi:")
    for path in csv_paths:
        print(f" - {path.parent.name} ({path.parent.resolve()})")

    # Verileri saklamak için sözlükler
    model_data = {}
    summary_metrics = []

    architectures = {
        "YOLO11N": "YOLO11 (Nano CNN - Tek Aşamalı)",
        "YOLO11S": "YOLO11 (Small CNN - Tek Aşamalı)",
        "YOLO11M": "YOLO11 (Medium CNN - Tek Aşamalı)",
        "YOLOV8S": "YOLOv8 (Small CNN - Tek Aşamalı)",
        "RTDETR-L": "RT-DETR (Large Transformer - Real-Time)"
    }

    # 2. CSV Dosyalarını Oku ve Parse Et
    for csv_path in csv_paths:
        raw_name = csv_path.parent.name.replace("_advanced", "").upper()
        model_name = raw_name
        
        try:
            df = pd.read_csv(csv_path)
            # Sütun isimlerindeki boşlukları temizle
            df.columns = [c.strip() for c in df.columns]
            
            # Sütun eşleşmelerini güvenli hale getir
            epoch_col = 'epoch'
            map50_col = [c for c in df.columns if 'mAP50(B)' in c or 'mAP50' in c and 'mAP50-95' not in c]
            map95_col = [c for c in df.columns if 'mAP50-95' in c]
            precision_col = [c for c in df.columns if 'precision' in c]
            recall_col = [c for c in df.columns if 'recall' in c]
            val_box_loss_col = [c for c in df.columns if 'val/box_loss' in c or 'val/box' in c or 'val/giou_loss' in c or 'val/giou' in c]
            train_box_loss_col = [c for c in df.columns if 'train/box_loss' in c or 'train/box' in c or 'train/giou_loss' in c or 'train/giou' in c]

            if not map50_col or not epoch_col:
                print(f"[HATA] {model_name} için gerekli metrik sütunları bulunamadı! Atlanıyor.")
                continue

            m50 = map50_col[0]
            m95 = map95_col[0] if map95_col else None
            prec = precision_col[0] if precision_col else None
            rec = recall_col[0] if recall_col else None
            vbox = val_box_loss_col[0] if val_box_loss_col else None
            tbox = train_box_loss_col[0] if train_box_loss_col else None

            # Verileri kaydet
            model_data[model_name] = {
                'epochs': df[epoch_col].values,
                'map50': df[m50].values,
                'map95': df[m95].values if m95 else np.zeros(len(df)),
                'val_box_loss': df[vbox].values if vbox else np.zeros(len(df)),
                'train_box_loss': df[tbox].values if tbox else np.zeros(len(df)),
            }

            # En iyi sonuçları özet tabloya ekle
            best_epoch_idx = df[m50].idxmax()
            best_map50 = df[m50].max()
            best_map95 = df[m95].iloc[best_epoch_idx] if m95 else 0.0
            best_prec = df[prec].iloc[best_epoch_idx] if prec else 0.0
            best_rec = df[rec].iloc[best_epoch_idx] if rec else 0.0
            total_epochs = len(df)

            summary_metrics.append({
                'Model': model_name,
                'Mimari Türü': architectures.get(model_name, "Özel Derin Öğrenme Modeli"),
                'Eğitilen Epoch': total_epochs,
                'En İyi mAP@50': f"{best_map50:.4f}",
                'mAP@50-95 (Best)': f"{best_map95:.4f}",
                'Precision (Best)': f"{best_prec:.4f}",
                'Recall (Best)': f"{best_rec:.4f}"
            })

        except Exception as e:
            print(f"[HATA] {csv_path} dosyası okunurken hata oluştu: {e}")

    if not model_data:
        print("[HATA] Analiz edilecek geçerli veri bulunamadı!")
        return

    # 3. Grafik Çizimi (Matplotlib Modern & Premium Stil)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Renk Paleti
    colors = ['#6C5CE7', '#00CEC9', '#FF7675', '#FDCB6E', '#0984E3', '#2D3436']

    for idx, (model_name, data) in enumerate(model_data.items()):
        color = colors[idx % len(colors)]
        
        # 1. Grafik: mAP@50 Doğruluk Eğrileri
        lbl = f"{model_name} ({architectures.get(model_name, 'Custom')})"
        axes[0].plot(data['epochs'], data['map50'], label=lbl, 
                     linewidth=2.5, color=color, alpha=0.9)
        
        # 2. Grafik: Box Kayıp Eğrileri
        axes[1].plot(data['epochs'], data['val_box_loss'], label=f"{model_name} (Val Loss)", 
                     linewidth=2.5, color=color, alpha=0.9)
        axes[1].plot(data['epochs'], data['train_box_loss'], linestyle='--', 
                     linewidth=1.2, color=color, alpha=0.5)

    # 1. Grafik Ayarları
    axes[0].set_title("Modellerin Doğruluk ve Yakınsama (mAP@50) Grafiği", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight='bold', labelpad=8)
    axes[0].set_ylabel("Metrik Başarım Skoru", fontsize=11, fontweight='bold', labelpad=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=True, facecolor='white', shadow=True, fontsize=9)
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # 2. Grafik Ayarları
    axes[1].set_title("Farklı Mimarilerin Box Kayıp Değişimleri (Train vs Val)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight='bold', labelpad=8)
    axes[1].set_ylabel("Kayıp (Box Loss / Loss)", fontsize=11, fontweight='bold', labelpad=8)
    axes[1].legend(frameon=True, facecolor='white', shadow=True, fontsize=9)
    axes[1].grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("CNN vs. Transformer Voleybol Nesne Tespit Karşılaştırmaları", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Grafiği Kaydet
    output_img = PROJECT_DIR / "model_comparison.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[BAŞARI] Çoklu model karşılaştırma grafiği kaydedildi: {output_img}")
    # plt.show() # Headless/Terminal ortamında pencere açılmasını önlemek için pasifleştirildi

    # 4. Performans Tablosunu Yazdır
    summary_df = pd.DataFrame(summary_metrics)
    print("\n" + "="*80)
    print("📊 AKADEMİK ÇOKLU MİMARİ KARŞILAŞTIRMA TABLOSU:")
    print("="*80)
    print(summary_df.to_markdown(index=False))
    print("="*80 + "\n")

def print_training_instructions():
    print("\n" + "-"*50)
    print("💡 MODEL EĞİTİM ADIMLARI:")
    print("Modelleri kıyaslamak için sırasıyla eğitin:")
    print("\n1) Tek Aşamalı CNN (YOLOv8 / YOLO11) Eğitimi:")
    print("   python scripts/train.py --model yolo11s.pt --epochs 50")
    print("   python scripts/train.py --model yolov8s.pt --epochs 50")
    print("\n2) Transformer Tabanlı Model (RT-DETR) Eğitimi:")
    print("   python scripts/train.py --model rtdetr-l.pt --epochs 50")
    print("\n3) Karşılaştırma Grafikleri ve Raporlama:")
    print("   python scripts/compare_models.py")
    print("-"*50 + "\n")

if __name__ == "__main__":
    compare_trained_models()

