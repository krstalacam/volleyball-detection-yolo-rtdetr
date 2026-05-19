import os
import cv2
import time
import torch
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from collections import deque

# Flask Uygulama Ayarları
app = Flask(__name__, template_folder="templates", static_folder="static")

# Proje Yolları
PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_DIR / "uploads"
STATIC_RESULTS_FOLDER = PROJECT_DIR / "static" / "results"

# Klasörleri Güvenli Şekilde Oluştur
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
STATIC_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Bellek Hafızasında Model Önbelleği (Model Caching)
# Bu sayede her API isteğinde ağır model ağırlıklarını diskten yüklemek yerine RAM'den anında çalıştırırız.
MODEL_CACHE = {}

def get_weights_path(model_dir_name, filename="best.pt"):
    """
    Traine edilmiş weights dosyasının yolunu arar. 
    1. Temiz dizin arar: runs/volleyball_detection/
    2. Fallback dizini arar (YOLO Windows task önekli): runs/detect/runs/volleyball_detection/
    """
    path_primary = PROJECT_DIR / "runs" / "volleyball_detection" / model_dir_name / "weights" / filename
    if path_primary.exists():
        return path_primary
        
    path_fallback = PROJECT_DIR / "runs" / "detect" / "runs" / "volleyball_detection" / model_dir_name / "weights" / filename
    if path_fallback.exists():
        return path_fallback
        
    return None

def load_selected_model(model_name):
    """
    Seçilen nesne tespit modelini RAM'e yükler veya önbellekten çeker.
    """
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
        
    print(f"[BEYİN] Modeli yükleme isteği geldi: {model_name}")
    stem_name = Path(model_name).stem
    
    # 1. RT-DETR Kontrolü
    if "rtdetr" in model_name.lower():
        from ultralytics import RTDETR
        weights_path = get_weights_path(f"{stem_name}_advanced", "best.pt")
        if weights_path:
            model = RTDETR(weights_path)
            print(f"[BEYİN] Eğitilmiş RT-DETR weights başarıyla yüklendi: {weights_path}")
        else:
            model = RTDETR(model_name)
            print(f"[UYARI] Eğitilmiş RT-DETR weights bulunamadı! Ön eğitimli '{model_name}' yüklendi.")
            
        MODEL_CACHE[model_name] = ("ultralytics_rtdetr", model, None)
        return MODEL_CACHE[model_name]
        
    # 2. Standart YOLO Kontrolü (YOLO11, YOLOv8 vb.)
    else:
        from ultralytics import YOLO
        weights_path = get_weights_path(f"{stem_name}_advanced", "best.pt")
        if weights_path:
            model = YOLO(weights_path)
            print(f"[BEYİN] Eğitilmiş {model_name} weights başarıyla yüklendi: {weights_path}")
        else:
            model = YOLO(model_name)
            print(f"[UYARI] Eğitilmiş {model_name} weights bulunamadı! Ön eğitimli '{model_name}' yüklendi.")
            
        MODEL_CACHE[model_name] = ("ultralytics_yolo", model, None)
        return MODEL_CACHE[model_name]

@app.route("/")
def home():
    """Ana sayfayı (HTML Dashboard) yükler."""
    return render_template("index.html")

@app.route("/api/models")
def get_models():
    """Kullanılabilir model seçeneklerini döner."""
    return jsonify({
        "models": [
            {"id": "yolo11s.pt", "name": "YOLO11s (Modern Tek Aşamalı CNN)", "recommended": True},
            {"id": "yolov8s.pt", "name": "YOLOv8s (Klasik Tek Aşamalı CNN)", "recommended": False},
            {"id": "rtdetr-l.pt", "name": "RT-DETR-L (Gelişmiş Transformer)", "recommended": False}
        ]
    })

@app.route("/api/predict-image", methods=["POST"])
def predict_image():
    """Görüntü analiz isteği."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Dosya gönderilmedi!"}), 400
        
    file = request.files["file"]
    model_id = request.form.get("model", "yolo11s.pt")
    conf = float(request.form.get("conf", 0.50))
    iou = float(request.form.get("iou", 0.45))
    
    if file.filename == "":
        return jsonify({"success": False, "error": "Geçersiz dosya ismi!"}), 400
        
    # Görüntüyü kaydet
    filename = f"img_{int(time.time())}_{file.filename}"
    file_path = UPLOAD_FOLDER / filename
    file.save(str(file_path))
    
    # Modeli Yükle / Önbellekten Getir
    try:
        model_type, model, device = load_selected_model(model_id)
    except Exception as e:
        return jsonify({"success": False, "error": f"Model yüklenirken hata oluştu: {str(e)}"}), 500
        
    start_time = time.time()
    annotated_filename = f"res_{filename}"
    annotated_path = STATIC_RESULTS_FOLDER / annotated_filename
    
    detected_count = 0
    
    # Ultralytics Modelleri (YOLO & RT-DETR) Tahminleme
    results = model.predict(
        source=str(file_path),
        conf=conf,
        iou=iou,
        device=device,
        verbose=False
    )
    res = results[0]
    detected_count = len(res.boxes)
    
    # r.plot() ile şık görselleştirme çizimi yap
    annotated_bgr = res.plot(conf=True, line_width=3, font_size=1.0, labels=True, boxes=True)
    cv2.imwrite(str(annotated_path), annotated_bgr)
        
    inference_time = int((time.time() - start_time) * 1000)
    
    return jsonify({
        "success": True,
        "result_url": f"/static/results/{annotated_filename}",
        "detected_count": detected_count,
        "inference_time_ms": inference_time
    })

@app.route("/api/predict-video", methods=["POST"])
def predict_video():
    """Video takip ve yörünge görselleştirme isteği."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Dosya gönderilmedi!"}), 400
        
    file = request.files["file"]
    model_id = request.form.get("model", "yolo11s.pt")
    conf = float(request.form.get("conf", 0.50))
    iou = float(request.form.get("iou", 0.45))
    
    if file.filename == "":
        return jsonify({"success": False, "error": "Geçersiz dosya ismi!"}), 400
        
    # Videoyu kaydet
    filename = f"vid_{int(time.time())}_{file.filename}"
    file_path = UPLOAD_FOLDER / filename
    file.save(str(file_path))
    
    # Modeli Yükle
    try:
        model_type, model, device = load_selected_model(model_id)
    except Exception as e:
        return jsonify({"success": False, "error": f"Model yüklenirken hata oluştu: {str(e)}"}), 500
        
    # Video işlemcisini başlat
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return jsonify({"success": False, "error": "Video açılamadı!"}), 500
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    output_filename = f"tracked_{filename}"
    output_path = STATIC_RESULTS_FOLDER / output_filename
    
    # Web tarayıcılarında doğrudan izlenebilmesi için 'avc1' veya standard 'mp4v' yaz
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Yörünge Takibi Deque kuyruğu
    ball_trail = deque(maxlen=30)
    
    # İşlemi başlat
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_center = None
        
        # Ultralytics Modelleri (YOLO & RT-DETR) ByteTrack Takibi
        # model.track persist=True kullanarak takip id korur
        # RT-DETR de track metodunu destekler
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf,
            iou=iou,
            device=device,
            verbose=False
        )
        res = results[0]
        
        if res.boxes and res.boxes.id is not None:
            for box, track_id in zip(res.boxes.xyxy, res.boxes.id):
                x1, y1, x2, y2 = box.cpu().numpy()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                current_center = (cx, cy)
                
                radius = int((x2 - x1) / 2)
                cv2.circle(frame, (cx, cy), radius + 5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(frame, f"Ball ID: {int(track_id)}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break
                    
        # Yörünge kuyruğuna ekle
        if current_center is not None:
            ball_trail.append(current_center)
            
        # Yörünge izlerini çiz (Kuyruklu yıldız degrade efekti)
        for i in range(1, len(ball_trail)):
            if ball_trail[i - 1] is None or ball_trail[i] is None:
                continue
                
            thickness = int(np.sqrt(30 / float(i + 1)) * 2)
            thickness = max(1, thickness)
            
            ratio = float(i) / len(ball_trail)
            color_b = int(255 * (1 - ratio))
            color_r = int(255 * ratio)
            color_g = int(100 * ratio)
            
            cv2.line(frame, ball_trail[i - 1], ball_trail[i], (color_b, color_g, color_r), thickness + 1)
            
        out.write(frame)
        
    cap.release()
    out.release()
    
    return jsonify({
        "success": True,
        "result_url": f"/static/results/{output_filename}"
    })

@app.route("/api/predict-frame", methods=["POST"])
def predict_frame():
    """
    Bireysel video karesi üzerinde diske yazmadan doğrudan RAM'de hızlı tahminleme yapar.
    Arayüzün canlı (live frame-by-frame) video takibi yapabilmesi için tasarlanmıştır.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Kare verisi gönderilmedi!"}), 400
        
    file = request.files["file"]
    model_id = request.form.get("model", "yolo11s.pt")
    conf = float(request.form.get("conf", 0.50))
    iou = float(request.form.get("iou", 0.45))
    
    # Resmi diske kaydetmeden doğrudan RAM'den oku (blazing-fast)
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return jsonify({"success": False, "error": "Geçersiz kare pikselleri!"}), 400
        
    try:
        model_type, model, device = load_selected_model(model_id)
    except Exception as e:
        return jsonify({"success": False, "error": f"Model yüklenirken hata: {str(e)}"}), 500
        
    start_time = time.time()
    boxes_out = []
    
    # A. Faster R-CNN (Saf PyTorch) Tahminleme
    if model_type == "pytorch_fasterrcnn":
        from torchvision.transforms import v2 as transforms
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            predictions = model(img_tensor)[0]
            
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= conf and label == 1: # Sınıf 1 -> ball
                x1, y1, x2, y2 = box
                boxes_out.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(score),
                    "class": "ball"
                })
                break # En yüksek skora sahip topu al
                
    # B. Ultralytics Modelleri (YOLO & RT-DETR) Tahminleme
    else:
        results = model.predict(
            source=img_bgr,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False
        )
        res = results[0]
        
        if res.boxes:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].cpu().item()
                boxes_out.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(score),
                    "class": "ball"
                })
                
    inference_time = int((time.time() - start_time) * 1000)
    
    return jsonify({
        "success": True,
        "boxes": boxes_out,
        "inference_time_ms": inference_time
    })

TRAINING_STATUS = {
    "running": False,
    "model_id": None,
    "message": "Hazir",
    "target_epochs": 0,
    "current_epoch": 0,
    "progress_pct": 0,
}
MODEL_REGISTRY = [
    {"id": "yolo11s.pt", "name": "YOLO11s"},
    {"id": "yolov8s.pt", "name": "YOLOv8s"},
    {"id": "rtdetr-l.pt", "name": "RT-DETR-L"},
]

def _is_model_trained(model_id):
    stem_name = Path(model_id).stem
    return get_weights_path(f"{stem_name}_advanced", "best.pt") is not None

def _get_training_results_csv(model_id):
    run_name = f"{Path(model_id).stem}_advanced"

    primary = PROJECT_DIR / "runs" / "volleyball_detection" / run_name / "results.csv"
    if primary.exists():
        return primary
    fallback = PROJECT_DIR / "runs" / "detect" / "runs" / "volleyball_detection" / run_name / "results.csv"
    if fallback.exists():
        return fallback
    return None

def _clean_training_directory(model_id):
    """
    Egit veya Yeniden Egit denildiginde, eski egitim dosyalarini tamamen siler.
    """
    run_name = f"{Path(model_id).stem}_advanced"
        
    dirs_to_delete = [
        PROJECT_DIR / "runs" / "volleyball_detection" / run_name,
        PROJECT_DIR / "runs" / "detect" / "runs" / "volleyball_detection" / run_name
    ]
    for d in dirs_to_delete:
        if d.exists():
            print(f"[TEMIZLIK] Eski egitim dizini siliniyor: {d}")
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(f"[UYARI] Dizin silinirken hata olustu: {d}, hata: {e}")

def _update_training_progress():
    if not TRAINING_STATUS["running"] or not TRAINING_STATUS["model_id"]:
        return
    csv_path = _get_training_results_csv(TRAINING_STATUS["model_id"])
    if not csv_path:
        return
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if len(lines) <= 1:
            return
        current_epoch = len(lines) - 1
        if float(TRAINING_STATUS.get("current_epoch", 0)) > current_epoch:
            return
        TRAINING_STATUS["current_epoch"] = current_epoch
        target = max(1, int(TRAINING_STATUS["target_epochs"]))
        pct = int(min(100, (current_epoch / target) * 100))
        TRAINING_STATUS["progress_pct"] = pct
        TRAINING_STATUS["message"] = f"{TRAINING_STATUS['model_id']} egitiliyor... ({current_epoch}/{target})"
    except Exception:
        pass

def _run_training_job(model_id, epochs, batch, patience, imgsz):
    # Egitim baslamadan once eski dosyalari temizle ve cache'i pop et
    _clean_training_directory(model_id)
    MODEL_CACHE.pop(model_id, None)

    TRAINING_STATUS["running"] = True
    TRAINING_STATUS["model_id"] = model_id
    TRAINING_STATUS["target_epochs"] = int(epochs)
    TRAINING_STATUS["current_epoch"] = 0
    TRAINING_STATUS["progress_pct"] = 1
    TRAINING_STATUS["message"] = f"{model_id} egitime hazirlaniyor..."
    try:
        def _epoch_progress_cb(current_epoch, target_epoch):
            if not TRAINING_STATUS["running"]:
                return
            current = float(current_epoch)
            TRAINING_STATUS["current_epoch"] = round(current, 2)
            target = max(1, int(target_epoch))
            TRAINING_STATUS["target_epochs"] = target
            TRAINING_STATUS["progress_pct"] = int(max(1, min(99, (current / target) * 100)))
            shown_current = min(target, max(0, current))
            TRAINING_STATUS["message"] = f"{model_id} egitiliyor... ({shown_current:.2f}/{target})"

        from scripts.train import run_training
        run_training(
            model_name=model_id,
            epochs=epochs,
            batch=batch,
            patience=patience,
            imgsz=imgsz,
            progress_callback=_epoch_progress_cb,
        )
        
        # Once running'i False yap ki, concurrent durum istekleri csv'den eski degerleri okumasin
        TRAINING_STATUS["running"] = False
        TRAINING_STATUS["current_epoch"] = TRAINING_STATUS["target_epochs"]
        TRAINING_STATUS["progress_pct"] = 100
        TRAINING_STATUS["message"] = "Egitim bitti."
    except Exception as e:
        TRAINING_STATUS["running"] = False
        TRAINING_STATUS["message"] = f"Egitim hatasi: {str(e)}"
    finally:
        TRAINING_STATUS["running"] = False

@app.route("/api/train/models")
def api_train_models():
    models = []
    for model in MODEL_REGISTRY:
        models.append({**model, "trained": _is_model_trained(model["id"])})
    return jsonify({"success": True, "models": models})

@app.route("/api/train/status")
def api_train_status():
    _update_training_progress()
    return jsonify({"success": True, **TRAINING_STATUS})

@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    if TRAINING_STATUS["running"]:
        return jsonify({"success": False, "error": "Su anda bir egitim calisiyor."}), 409

    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model")
    epochs = int(payload.get("epochs", 30))
    batch = int(payload.get("batch", 8))
    patience = int(payload.get("patience", 10))
    imgsz = int(payload.get("imgsz", 640))

    valid_ids = {m["id"] for m in MODEL_REGISTRY}
    if model_id not in valid_ids:
        return jsonify({"success": False, "error": "Gecersiz model secimi."}), 400

    import threading
    thread = threading.Thread(
        target=_run_training_job,
        args=(model_id, epochs, batch, patience, imgsz),
        daemon=True,
    )
    thread.start()
    return jsonify({"success": True, "message": f"{model_id} egitimi baslatildi."})

@app.after_request
def add_header(r):
    """Tarayıcının dinamik API yanıtlarını önbelleğe almasını önler."""
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

if __name__ == "__main__":
    # Flask sunucuyu local olarak 5000 portunda başlat
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
