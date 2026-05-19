import os
import cv2
import hashlib
import shutil
from pathlib import Path

# Proje Kök Dizini ve Yollar
PROJECT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_DIR / "uploads"
LABELING_DIR = PROJECT_DIR / "manual_labeling"
IMAGES_OUT_DIR = LABELING_DIR / "images"
LABELS_OUT_DIR = LABELING_DIR / "labels"

def get_md5(file_path):
    """Dosyanın MD5 hash değerini hesaplar."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def prepare_dataset():
    print("=" * 60)
    print("MANUEL ETIKETLEME ICIN VERI SETI HAZIRLANIYOR")
    print("=" * 60)

    # 1. Klasörleri Oluştur
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[KLASOR] Klasorler basariyla hazirlandi: {UPLOAD_DIR} ve {LABELING_DIR}")

    # 2. Uploads Klasöründeki Eşsiz Görüntüleri Kopyala
    unique_hashes = set()
    copied_image_count = 0
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    if UPLOAD_DIR.exists():
        print(f"[BUL] '{UPLOAD_DIR}' taraniyor...")
        for p in UPLOAD_DIR.iterdir():
            if p.is_file() and p.suffix.lower() in valid_extensions:
                # Eşsizlik kontrolü (MD5 ile kopya dosyaları engelleme)
                try:
                    file_hash = get_md5(p)
                    if file_hash not in unique_hashes:
                        unique_hashes.add(file_hash)
                        # Dosyayı kopyala
                        dest_name = f"manual_{p.name}"
                        shutil.copy2(p, IMAGES_OUT_DIR / dest_name)
                        copied_image_count += 1
                except Exception as e:
                    print(f"[UYARI] {p.name} islenirken hata olustu: {e}")
                    
        print(f"[BASARI] {copied_image_count} adet essiz gorsel '{IMAGES_OUT_DIR}' klasorune aktarildi.")
    else:
        print(f"[UYARI] '{UPLOAD_DIR}' klasoru bulunamadi!")

    # 3. Uploads Klasöründeki Videolardan Kare Çıkar
    extracted_frame_count = 0
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    
    if UPLOAD_DIR.exists():
        video_files = [p for p in UPLOAD_DIR.iterdir() if p.is_file() and p.suffix.lower() in video_extensions]
        
        if video_files:
            # En son veya en büyük videoyu seçelim (veya hepsini)
            # Genelde WhatsApp videosu tek bir videodur, ilkini işleyelim
            selected_video = video_files[0]
            print(f"[VIDEO] '{selected_video.name}' videosundan kareler cikariliyor...")
            
            # Eski çıkartılmış frame dosyalarını ve etiketlerini temizleyelim (çakışma önleme)
            print("[TEMIZLIK] Eski 'frame_' resimleri ve etiketleri temizleniyor...")
            for old_file in IMAGES_OUT_DIR.glob("frame_*.jpg"):
                try:
                    old_file.unlink()
                except Exception:
                    pass
            for old_label in LABELS_OUT_DIR.glob("frame_*.txt"):
                try:
                    old_label.unlink()
                except Exception:
                    pass
            
            cap = cv2.VideoCapture(str(selected_video))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if cap.isOpened() and total_frames > 0:
                # Toplam 100 kare çıkaralım
                num_frames_to_extract = 100
                interval = max(1, total_frames // num_frames_to_extract)
                
                frame_idx = 0
                extracted_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_idx % interval == 0 and extracted_idx < num_frames_to_extract:
                        frame_name = f"frame_{selected_video.stem}_{extracted_idx:04d}.jpg"
                        frame_path = IMAGES_OUT_DIR / frame_name
                        
                        # Windows Unicode (Türkçe karakterli yol) için güvenli kayıt yöntemi
                        is_success, im_buf_arr = cv2.imencode(".jpg", frame)
                        if is_success:
                            with open(frame_path, "wb") as f:
                                f.write(im_buf_arr.tobytes())
                            extracted_frame_count += 1
                        extracted_idx += 1
                        
                    frame_idx += 1
                cap.release()
                print(f"[BASARI] Videodan {extracted_frame_count} adet kare '{IMAGES_OUT_DIR}' klasorune aktarildi.")
            else:
                print(f"[HATA] Video acilamadi: {selected_video.name}")

    # 4. Sınıf Dosyasını (classes.txt) Oluştur
    # labelImg'in etiket isimlerini otomatik tanıması için classes.txt gereklidir
    classes_content = "ball\n"
    
    # Hem images hem labels içerisine classes.txt yazalım (labelImg için ideal)
    with open(IMAGES_OUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
        f.write(classes_content)
    with open(LABELS_OUT_DIR / "classes.txt", "w", encoding="utf-8") as f:
        f.write(classes_content)
        
    print("[SINIF] 'classes.txt' dosyalari basariyla olusturuldu (Sinif: 'ball').")
    print("\n" + "=" * 60)
    print("HAZIRLIK TAMAMLANDI!")
    print(f"Toplam Gorsel Sayisi: {copied_image_count + extracted_frame_count}")
    print("Simdi manuel etiketleme yapmaya hazirsiniz!")
    print("=" * 60)

if __name__ == "__main__":
    prepare_dataset()
