import os
import shutil
import random
from pathlib import Path

# Proje Kök Dizini ve Yollar
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
LABELING_DIR = PROJECT_DIR / "manual_labeling"
LABELING_IMAGES = LABELING_DIR / "images"
LABELING_LABELS = LABELING_DIR / "labels"

# Master (Ana Kaynak) Veri Tabanı Yolları
BACKUP_DIR = PROJECT_DIR / "backupdata"
BACKUP_TRAIN_IMAGES = BACKUP_DIR / "train" / "images"
BACKUP_TRAIN_LABELS = BACKUP_DIR / "train" / "labels"
BACKUP_VALID_IMAGES = BACKUP_DIR / "valid" / "images"
BACKUP_VALID_LABELS = BACKUP_DIR / "valid" / "labels"

# Aktif Eğitim ve Doğrulama Hedef Yolları
TRAIN_IMAGES = DATA_DIR / "train" / "images"
TRAIN_LABELS = DATA_DIR / "train" / "labels"
VALID_IMAGES = DATA_DIR / "valid" / "images"
VALID_LABELS = DATA_DIR / "valid" / "labels"

def get_unique_filename(target_dir, filename):
    """Hedef klasörde aynı isimde dosya varsa, ismin sonuna benzersiz bir sayaç ekler."""
    path = Path(target_dir) / filename
    if not path.exists():
        return filename
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while (Path(target_dir) / f"{stem}_{counter}{suffix}").exists():
        counter += 1
    return f"{stem}_{counter}{suffix}"

def merge_and_split_dataset(val_ratio=0.20):
    print("=" * 60)
    print("MASTER VERI TABANI BIRLESTIRME VE YENIDEN BOLME SISTEMI")
    print("=" * 60)

    # 1. Master klasörlerin varlığından emin ol
    BACKUP_TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    BACKUP_TRAIN_LABELS.mkdir(parents=True, exist_ok=True)
    BACKUP_VALID_IMAGES.mkdir(parents=True, exist_ok=True)
    BACKUP_VALID_LABELS.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # 2. Yeni Etiketlenen Görselleri Tara ve Kalıcı Olarak Master'a (backupdata) Arşivle
    archived_count = 0
    if LABELING_IMAGES.exists():
        print("[TARAMA] Yeni etiketlenmis gorseller taranıyor...")
        for img_path in list(LABELING_IMAGES.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in img_extensions:
                # classes.txt dosyasını atla
                if img_path.name == "classes.txt":
                    continue
                    
                # Etiket dosyasını ara
                lbl_path_1 = LABELING_LABELS / f"{img_path.stem}.txt"
                lbl_path_2 = LABELING_IMAGES / f"{img_path.stem}.txt"
                
                lbl_path = None
                if lbl_path_1.exists():
                    lbl_path = lbl_path_1
                elif lbl_path_2.exists():
                    lbl_path = lbl_path_2
                    
                if lbl_path is not None:
                    # Doğrudan senkronizasyon (aynı isimde varsa üzerine yazar, kopya oluşturmaz)
                    dest_img_path = BACKUP_TRAIN_IMAGES / img_path.name
                    dest_lbl_path = BACKUP_TRAIN_LABELS / lbl_path.name
                    
                    try:
                        # 1. Kalıcı olarak backupdata/train klasörüne kopyala (senkronize et)
                        shutil.copy2(img_path, dest_img_path)
                        shutil.copy2(lbl_path, dest_lbl_path)
                        
                        # Artık manual_labeling/ klasöründeki dosyaları silmiyoruz!
                        archived_count += 1
                        print(f"[SENKRONIZE] '{img_path.name}' -> Kalici master veri tabanina aktarildi (manual_labeling klasorunde de korundu).")
                    except Exception as e:
                        print(f"[UYARI] '{img_path.name}' eslesirken hata olustu: {e}")

        if archived_count > 0:
            print(f"[BASARI] {archived_count} adet yeni veri cifti kalici olarak 'backupdata/' klasorune arşivlendi.")
        else:
            print("[BILGI] Manuel etiketleme klasöründe yeni etiketlenmiş görsel bulunamadı.")

    # 3. Havuz Oluştur ve Master (backupdata) İçindeki Tüm Verileri Tara
    dataset_pool = {}
    
    # A. Master Train Klasörünü Tara
    backup_train_count = 0
    for img_path in BACKUP_TRAIN_IMAGES.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in img_extensions:
            lbl_path = BACKUP_TRAIN_LABELS / f"{img_path.stem}.txt"
            if lbl_path.exists():
                dataset_pool[img_path.name] = (img_path, lbl_path)
                backup_train_count += 1
                
    # B. Master Valid Klasörünü Tara
    backup_valid_count = 0
    for img_path in BACKUP_VALID_IMAGES.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in img_extensions:
            lbl_path = BACKUP_VALID_LABELS / f"{img_path.stem}.txt"
            if lbl_path.exists():
                dataset_pool[img_path.name] = (img_path, lbl_path)
                backup_valid_count += 1

    total_pairs = len(dataset_pool)
    print(f"[MASTER TARAMA] Master veri tabanında (backupdata) toplam {total_pairs} adet geceli (resim+etiket) cift bulundu.")
    print(f"  - Kaynak Train Çifti: {backup_train_count}")
    print(f"  - Kaynak Valid Çifti: {backup_valid_count}")
    
    if total_pairs == 0:
        print("[HATA] Master veri tabanında hiç resim ve etiket çifti bulunamadı! Birleşme iptal edildi.")
        return

    # 4. Görüntüleri ve Etiketleri Güvenli Bir Şekilde Staging Klasörüne Kopyala (Veri Kaybı Koruması)
    temp_staging_dir = DATA_DIR / "temp_staging"
    temp_staging_images = temp_staging_dir / "images"
    temp_staging_labels = temp_staging_dir / "labels"
    
    if temp_staging_dir.exists():
        shutil.rmtree(temp_staging_dir)
    temp_staging_images.mkdir(parents=True, exist_ok=True)
    temp_staging_labels.mkdir(parents=True, exist_ok=True)
    
    print("[STAGING] Dosyalar gecici staging alanina kopyalaniyor (Veri kaybi korumasi aktif)...")
    staged_pairs = []
    for img_name, (img_src, lbl_src) in dataset_pool.items():
        img_staged = temp_staging_images / img_src.name
        lbl_staged = temp_staging_labels / lbl_src.name
        
        try:
            shutil.copy2(img_src, img_staged)
            shutil.copy2(lbl_src, lbl_staged)
            staged_pairs.append((img_staged, lbl_staged))
        except Exception as e:
            print(f"[UYARI] {img_name} staging alanina kopyalanirken hata olustu: {e}")
         
    # 5. Verileri Karıştır ve Böl
    random.seed(42)  # Tekrarlanabilir bölme için
    random.shuffle(staged_pairs)
    
    val_size = int(total_pairs * val_ratio)
    val_pairs = staged_pairs[:val_size]
    train_pairs = staged_pairs[val_size:]
    
    print(f"[BOLME] Veri seti bölünüyor: {len(train_pairs)} Train (%{100 - int(val_ratio*100)}), {len(val_pairs)} Valid (%{int(val_ratio*100)})")

    # 6. Eski Aktif Klasörleri Güvenle Temizle ve Yeni Yapıyı Oluştur
    for folder in [TRAIN_IMAGES, TRAIN_LABELS, VALID_IMAGES, VALID_LABELS]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    # 7. Dosyaları Staging Alanından Hedef Klasörlere Kopyala
    print("[KOPYALAMA] Dosyalar aktif egitim klasorlerine dagitiliyor...")
    
    # Train Kopyalama
    for img_src, lbl_src in train_pairs:
        shutil.copy2(img_src, TRAIN_IMAGES / img_src.name)
        shutil.copy2(lbl_src, TRAIN_LABELS / lbl_src.name)
        
    # Valid Kopyalama
    for img_src, lbl_src in val_pairs:
        shutil.copy2(img_src, VALID_IMAGES / img_src.name)
        shutil.copy2(lbl_src, VALID_LABELS / lbl_src.name)

    # 8. Gecici Staging Klasörünü Temizle
    if temp_staging_dir.exists():
        shutil.rmtree(temp_staging_dir)
        
    print("[KOPYALAMA] Tum dosyalar basariyla yerlestirildi ve gecici staging temizlendi.")

    # 9. data.yaml Dosyasını Güncelle
    yaml_path = DATA_DIR / "data.yaml"
    data_path_str = str(DATA_DIR).replace("\\", "/")
    
    yaml_content = f"""path: {data_path_str}
train: train/images
val: valid/images

nc: 1
names: ['ball']
"""
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"[YAML] 'data.yaml' dosyası güncellendi: {yaml_path}")
    print("\n" + "=" * 60)
    print("[TEBRIKLER] ISLEM BASARIYLA TAMAMLANDI!")
    print(f"Master Veri Tabanı (backupdata) Toplamı: {total_pairs} Gorsel Çifti")
    print(f"Yeni Aktif Eğitim Seti: {len(train_pairs)} Resim + Etiket")
    print(f"Yeni Aktif Doğrulama Seti: {len(val_pairs)} Resim + Etiket")
    print("Artik yeni model egitimini baslatabilirsiniz!")
    print("=" * 60)

if __name__ == "__main__":
    merge_and_split_dataset(val_ratio=0.20)
