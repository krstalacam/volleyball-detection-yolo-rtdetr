import os
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# Proje Kök Dizini ve Yolları
PROJECT_DIR = Path(__file__).resolve().parent.parent

def get_weights_path(model_dir_name, filename="best.pt"):
    """Trained weights dosyasını arar (Temiz absolute veya YOLO task-prefixed relative)."""
    path_primary = PROJECT_DIR / "runs" / "volleyball_detection" / model_dir_name / "weights" / filename
    if path_primary.exists():
        return path_primary
    path_fallback = PROJECT_DIR / "runs" / "detect" / "runs" / "volleyball_detection" / model_dir_name / "weights" / filename
    if path_fallback.exists():
        return path_fallback
    return None

# Varsayılan Ağırlık Yolu Arama
DEFAULT_WEIGHTS_PATH = get_weights_path("yolo11s_advanced", "best.pt")

def track_volleyball(video_path, weights_path=None, output_path=None, conf=0.15, iou=0.45, max_trail_len=30):
    """
    Video üzerinde voleybol topunu takip eder, topun yörüngesini (trajectory trail) 
    görselleştirerek yeni bir video dosyası olarak kaydeder.
    """
    print("="*60)
    print("🎥 VOLEYBOL TOPU TAKİBİ VE YÖRÜNGE GÖRSELLEŞTİRME (TRACKING) 🎥")
    print("="*60)

    # Ağırlık yolunu kararlaştır
    if weights_path is None:
        weights_path = DEFAULT_WEIGHTS_PATH if DEFAULT_WEIGHTS_PATH else "yolo11s.pt"
        
    weights_path = Path(weights_path)
    if not weights_path.exists() and weights_path.name != "yolo11s.pt":
        # Eğer özel yol verilmiş ama yoksa fallback aramayı dene
        resolved = get_weights_path(weights_path.parent.parent.name, weights_path.name)
        if resolved:
            weights_path = resolved
            
    if not weights_path.exists():
        print(f"[UYARI] Eğitilmiş '{weights_path}' ağırlığı bulunamadı!")
        fallback_model = "yolo11s.pt"
        print(f"[BİLGİ] Varsayılan ön eğitimli '{fallback_model}' ağırlığı kullanılarak takip yapılacak.")
        model = YOLO(fallback_model)
    else:
        print(f"[BİLGİ] Eğitilmiş model yüklendi: {weights_path}")
        model = YOLO(weights_path)

    # 2. Video Giriş Kontrolü
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[HATA] Giriş videosu bulunamadı: {video_path}")
        print("[İPUCU] Takip işlemini test etmek için lütfen bir voleybol maç videosu (.mp4) ekleyin.")
        return

    # Video yakalayıcıyı başlat
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[HATA] Video dosyası açılamadı: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VİDEO BİLGİSİ] Genişlik: {width}px, Yükseklik: {height}px, FPS: {fps:.2f}, Toplam Kare: {total_frames}")

    # Çıktı video yolu ayarları
    if output_path is None:
        output_path = PROJECT_DIR / f"tracked_{video_path.name}"
    else:
        output_path = Path(output_path)

    # Video yazıcıyı tanımla (MP4V veya XVID kodek)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # 3. Yörünge Takibi Belleği (Deque)
    # Topun geçmişteki koordinatlarını tutmak için kuyruk yapısı kullanıyoruz
    ball_trail = deque(maxlen=max_trail_len)

    # Cihaz belirleme
    device = 0 if torch.cuda.is_available() else "cpu"

    print("[TAKİP] Video kare kare işleniyor ve top takip ediliyor...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # YOLO11 ByteTrack Takipçisini Çalıştır
        # persist=True -> Bir kareden diğerine takip kimliğini (ID) korur.
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml", # ByteTrack, hızlı nesneler için mükemmeldir
            conf=conf,
            iou=iou,
            device=device,
            verbose=False
        )
        
        # Çıkarım sonucunu al
        res = results[0]
        
        # Topun mevcut karedeki merkez noktasını bul
        current_center = None
        
        # Eğer tespit ve takip kimlikleri varsa
        if res.boxes and res.boxes.id is not None:
            # Kutuları çiz
            for box, track_id in zip(res.boxes.xyxy, res.boxes.id):
                x1, y1, x2, y2 = box.cpu().numpy()
                track_id = int(track_id.cpu().item())
                
                # Topun merkez koordinatlarını hesapla
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                current_center = (cx, cy)
                
                # Topun etrafına şık bir daire çiz
                radius = int((x2 - x1) / 2)
                cv2.circle(frame, (cx, cy), radius + 5, (0, 255, 0), 2) # Yeşil dış halka
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)        # Kırmızı merkez noktası
                
                # Takip ID'sini ekrana yazdır
                cv2.putText(frame, f"Ball ID: {track_id}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # İlk tespit edilen topu ana nesne kabul et ve döngüden çık (Genelde 1 top bulunur)
                break
                
        # Eğer top tespit edildiyse yörünge kuyruğuna ekle, edilmediyse kuyruğun en eskisini yavaşça temizle
        if current_center is not None:
            ball_trail.append(current_center)
        elif len(ball_trail) > 0:
            # Topun kaybolduğu karelerde yörüngenin birden yok olmaması için 
            # en eski elemanı çıkararak yumuşak bir sönümleme yapabiliriz
            pass

        # 4. Yörünge Çizgisini Çiz (Trajectory Trail)
        # Çizginin kalınlığı ve opaklığı geçmişten bugüne doğru artar (kuyruklu yıldız efekti)
        for i in range(1, len(ball_trail)):
            if ball_trail[i - 1] is None or ball_trail[i] is None:
                continue
                
            # Geçmişe göre çizgi kalınlığını ayarla (son noktalara doğru kalınlaşır)
            thickness = int(np.sqrt(max_trail_len / float(i + 1)) * 2)
            thickness = max(1, thickness)
            
            # Renk geçişi (Maviden Kırmızıya modern bir yörünge efekti)
            # i/len(ball_trail) oranı ile renk interpolasyonu
            ratio = float(i) / len(ball_trail)
            color_b = int(255 * (1 - ratio))
            color_r = int(255 * ratio)
            color_g = int(100 * ratio)
            
            cv2.line(frame, ball_trail[i - 1], ball_trail[i], (color_b, color_g, color_r), thickness + 1)
            
        # İlerleme durumunu terminale yazdır
        if frame_count % 50 == 0 or frame_count == total_frames:
            print(f"[İLERLEME] İşlenen Kare: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
            
        # Kareyi çıktı videosuna yaz
        out.write(frame)

    # Kaynakları serbest bırak
    cap.release()
    out.release()
    print("\n" + "="*60)
    print("🎉 VİDEO TAKİBİ BAŞARIYLA TAMAMLANDI! 🎉")
    print(f"[BAŞARI] Takip videosu kaydedildi: {output_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Test çalıştırması (Eğer elinizde bir video varsa alt satırdaki yolu değiştirerek çalıştırabilirsiniz)
    dummy_video_path = PROJECT_DIR / "volleyball_match.mp4"
    if dummy_video_path.exists():
        track_volleyball(video_path=dummy_video_path, conf=0.15)
    else:
        print("[İPUCU] Video takibi yapabilmek için proje dizinine 'volleyball_match.mp4' adında bir video dosyası ekleyip bu scripti çalıştırabilirsiniz.")

