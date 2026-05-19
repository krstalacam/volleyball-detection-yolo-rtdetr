import os
import cv2
import yaml
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Proje Kök Dizini (Windows ve Linux uyumlu)
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

class DatasetVisualizer:
    """
    Voleybol Topu Tespit veri seti analizi ve görselleştirme sınıfı.
    Cross-platform uyumlu, hata toleranslı ve modern görsel tasarımlara sahiptir.
    """
    def __init__(self, data_dir=DATA_DIR, data_types=["train", "valid"]):
        self.data_dir = Path(data_dir)
        self.data_types = data_types
        self.yaml_path = self.data_dir / "data.yaml"
        
        # Grafik renk paletleri (Modern & Premium HSL tonları)
        self.colors = {
            "train": "#6C5CE7",   # Canlı Menekşe
            "valid": "#00CEC9",   # Canlı Turkuaz
            "test": "#FF7675"     # Yumuşak Kırmızı
        }
        
        self.class_dict = {}
        self.vis_datas = {}
        self.analysis_datas = {}
        self.im_paths = {}
        
        self._load_class_names()
        self._parse_dataset()

    def _load_class_names(self):
        """data.yaml dosyasından sınıf isimlerini güvenli bir şekilde yükler."""
        if not self.yaml_path.exists():
            print(f"[UYARI] {self.yaml_path} bulunamadı! Varsayılan ['ball'] sınıfı kullanılacak.")
            self.class_dict = {0: "ball"}
            return
            
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        class_names = data.get('names', ['ball'])
        if isinstance(class_names, list):
            self.class_dict = {idx: name for idx, name in enumerate(class_names)}
        elif isinstance(class_names, dict):
            self.class_dict = {int(k): v for k, v in class_names.items()}
        else:
            self.class_dict = {0: "ball"}
            
        print(f"[BİLGİ] Veri setindeki sınıflar: {self.class_dict}")

    def _parse_dataset(self):
        """Etiket dosyalarını ve görüntü yollarını güvenli şekilde parse eder."""
        for data_type in self.data_types:
            type_dir = self.data_dir / data_type
            img_dir = type_dir / "images"
            lbl_dir = type_dir / "labels"
            
            if not img_dir.exists():
                print(f"[UYARI] {img_dir} dizini bulunamadı! Atlanıyor...")
                continue
                
            # Desteklenen resim formatları
            valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
            img_paths = [p for p in img_dir.glob("*") if p.suffix.lower() in valid_extensions]
            
            bboxes_list = []
            analysis_counts = {name: 0 for name in self.class_dict.values()}
            valid_img_paths = []
            
            for im_path in img_paths:
                # Görüntüye ait YOLO etiket dosyasını bul (aynı isimli .txt)
                lbl_path = lbl_dir / f"{im_path.stem}.txt"
                bboxes = []
                
                if lbl_path.exists():
                    try:
                        with open(lbl_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                # Geçersiz etiket sınıf ID koruması
                                if class_id in self.class_dict:
                                    cls_name = self.class_dict[class_id]
                                    coords = [float(x) for x in parts[1:5]]
                                    bboxes.append([cls_name] + coords)
                                    analysis_counts[cls_name] += 1
                                    
                    except Exception as e:
                        print(f"[HATA] {lbl_path} okunurken hata oluştu: {e}")
                
                # Etiketsiz veya etiketli fark etmeksizin listeye ekle
                bboxes_list.append(bboxes)
                valid_img_paths.append(im_path)
                
            self.vis_datas[data_type] = bboxes_list
            self.analysis_datas[data_type] = analysis_counts
            self.im_paths[data_type] = valid_img_paths
            
            print(f"[BİLGİ] {data_type.upper()} alt kümesinde {len(valid_img_paths)} görüntü ve {sum(analysis_counts.values())} nesne bulundu.")

    def visualize_samples(self, data_type="train", n_ims=9, rows=3):
        """Rastgele görüntüleri üzerlerindeki YOLO bounding box'ları ile birlikte çizer."""
        if data_type not in self.vis_datas or not self.vis_datas[data_type]:
            print(f"[HATA] {data_type} görselleştirmek için veri bulunamadı!")
            return
            
        print(f"\n[GÖRSELLEŞTİRME] {data_type.upper()} veri kümesinden rastgele {n_ims} görsel çiziliyor...")
        
        indices = random.sample(range(len(self.vis_datas[data_type])), min(n_ims, len(self.vis_datas[data_type])))
        cols = int(np.ceil(n_ims / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
        axes = axes.flatten() if n_ims > 1 else [axes]
        
        # Farklı sınıflar için renk havuzu oluştur
        np.random.seed(42)
        box_colors = {}
        for cls in self.class_dict.values():
            box_colors[cls] = tuple(np.random.randint(50, 255, size=3).tolist())
            
        for i, idx in enumerate(indices):
            im_path = self.im_paths[data_type][idx]
            bboxes = self.vis_datas[data_type][idx]
            
            # Görüntüyü OpenCV ile oku
            img = cv2.imread(str(im_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _ = img.shape
            
            # Kutuları çiz
            for bbox in bboxes:
                cls_name, x_center, y_center, w, h = bbox
                
                # YOLO normalize formatı pixel koordinatlarına dönüştürme
                x_min = int((x_center - w / 2) * width)
                y_min = int((y_center - h / 2) * height)
                x_max = int((x_center + w / 2) * width)
                y_max = int((y_center + h / 2) * height)
                
                # Sınır taşmalarını engelle
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(width - 1, x_max), min(height - 1, y_max)
                
                # Kutu rengi
                color = box_colors[cls_name]
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
                
                # Etiket Metni ekleme
                label = f"{cls_name}"
                font_scale = 0.6
                thickness = 2
                (w_t, h_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(img, (x_min, y_min - h_t - 5), (x_min + w_t, y_min), color, -1)
                cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
            ax = axes[i]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Görsel #{idx} | {len(bboxes)} Top", fontsize=10, fontweight="bold")
            
        # Kullanılmayan subplot'ları gizle
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
            
        plt.tight_layout()
        plt.show()

    def analyze_class_distribution(self):
        """Veri kümesindeki sınıfların dağılımını modern bir bar grafik ile analiz eder."""
        print("\n[ANALİZ] Sınıf dağılımı analiz ediliyor...")
        
        # Mevcut verileri topla
        existing_types = [t for t in self.data_types if t in self.analysis_datas]
        if not existing_types:
            print("[HATA] Analiz edecek veri bulunamadı!")
            return
            
        classes = list(self.class_dict.values())
        
        # Modern stil ayarları
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(classes))
        width = 0.25
        
        for idx, d_type in enumerate(existing_types):
            counts = [self.analysis_datas[d_type].get(cls, 0) for cls in classes]
            offset = (idx - len(existing_types)/2 + 0.5) * width
            
            bars = ax.bar(x + offset, counts, width, label=f"{d_type.upper()} Kümesi", 
                          color=self.colors.get(d_type, "#000000"), edgecolor="white", linewidth=1.2, alpha=0.9)
            
            # Barların üzerine sayısal değerleri ekle
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2D3436')

        ax.set_xlabel("Sınıf İsimleri", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("Nesne Adedi", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_title("Voleybol Nesne Dağılımı ve Sınıf Dengesi Analizi", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Test Etme
    visualizer = DatasetVisualizer(data_dir=DATA_DIR, data_types=["train", "valid"])
    visualizer.analyze_class_distribution()
    visualizer.visualize_samples(data_type="train", n_ims=6, rows=2)
