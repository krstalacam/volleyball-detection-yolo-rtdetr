# 🏐 Voleybol Topu Tespit ve Takip Projesi

Bu proje, derin öğrenme yöntemleriyle (YOLO11 ve RT-DETR) voleybol topunu tespit etmeyi, takip etmeyi ve bu modelleri karşılaştırmayı amaçlayan bir okul projesidir. Ayrıca projeyi kolayca test edebilmek için şık bir Flask web arayüzü (kontrol paneli) de içerir.

---

## 📁 Proje Klasör Yapısı
* **`app.py`**: Web arayüzünü başlatan ana dosya.
* **`scripts/`**: Model eğitimleri, karşılaştırmalar, testler ve video takip kodlarının bulunduğu klasör.
* **`data/` & `backupdata/`**: Eğitilen voleybol topu veri setimiz (görseller ve etiketler).
* **`runs/`**: Modellerin eğitim sonuçları ve `.pt` ağırlık dosyaları.

---

## 🚀 Projeyi Çalıştırma Adımları

### 1. Sanal Ortamı Aktif Etme
Projedeki kütüphanelerin çalışması için terminalde (PowerShell) şu komutla sanal ortamı aktif etmeliyiz:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Bağımlılıkları ve GPU Desteğini Kurma (İlk Çalıştırmada)
Sanal ortamı aktif ettikten sonra, projede gerekli kütüphaneleri ve NVIDIA ekran kartınızın (GPU) kullanılmasını sağlayacak sürücü desteklerini kurmak için sırasıyla şu komutları çalıştırın:

* **NVIDIA GPU (CUDA) Desteğiyle PyTorch Kurulumu:**
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
  *(Not: CUDA desteğiyle PyTorch yüklemek eğitimlerin ve tahminlerin çok hızlı bir şekilde NVIDIA GPU üzerinde yapılmasını sağlar).*

* **Diğer Proje Bağımlılıklarının Kurulumu (`pyproject.toml` üzerinden):**
  ```powershell
  pip install .
  ```
  *(Bu komut `pyproject.toml` dosyasını okuyarak Flask, OpenCV, matplotlib vb. diğer tüm gerekli araçları otomatik kuracaktır).*

### 3. Web Arayüzünü Çalıştırma (En Kolayı 🌟)
Modelleri tarayıcıdan görsel olarak test etmek, fotoğraf/video yükleyip top tespitlerini ve takip yörüngelerini canlı görmek için:
```powershell
python app.py
```
Komutu çalıştırdıktan sonra tarayıcında **`http://127.0.0.1:5000`** adresine gitmen yeterli.

### 4. Yeni Veri Etiketleme (LabelImg)
Kendi çektiğiniz videolardan kare çıkartıp, bunları etiketlemek ve hazır veri setine sıfır kayıp riskiyle entegre etmek için bu basit kılavuzu takip edebilirsiniz:

#### 1️⃣ Videoyu Yükleyin
Etiketlemek istediğiniz videoyu (örn. `voleybol.mp4`) şu klasörün içerisine atın:
📂 **`uploads/`**

#### 2️⃣ Videodan Kareleri Çıkartın
Terminalde şu komutu çalıştırarak videodan otomatik 100 adet kare görsel üretin:
```powershell
python scripts/prepare_manual_dataset.py
```
*(Bu komut videoyu işler, kareleri `manual_labeling/images/` klasörüne çıkartır ve etiket sınıfını hazırlar).*

#### 3️⃣ Etiketleme Programını (labelImg) Çalıştırın
Terminalde şu komutla programı başlatın:
```powershell
python scripts/run_labelimg.py
```
##### ⚡ Yıldırım Hızında Etiketleme Akışı (Süper Güçler Aktif!):
Sizin için **Otomatik Kaydetme (Auto Save)** ve **Tek Sınıf Modu (Single Class)** özelliklerini varsayılan olarak aktifleştirdik!

1. **W Tuşuna basın.**
2. **Voleybol topunu kutulayın** (Sınıf seçme penceresi çıkmaz, doğrudan `ball` sınıfı atanır).
3. **D Tuşuna basın** (Resim arka planda otomatik kaydedilir ve anında sonraki resme geçer!).

*   *Yani her görsel için sadece **W ➔ Kutula ➔ D** yapmanız yeterlidir. Ctrl+S tuşuna basmanıza kesinlikle gerek yoktur!*
*   **A Tuşu:** Önceki resme geri döner.

#### 4️⃣ Etiketleri Master Veri Tabanı ile Birleştirin ve Yeniden Bölün (%80 Train / %20 Valid)
Etiketleme bittiğinde programı kapatın ve birleştirmeyi başlatın:
```powershell
python scripts/merge_and_split.py
```
*(Bu komut; yeni etiketlediğiniz görselleri **silmeden** doğrudan kalıcı olarak **`backupdata/` (Master Veri Tabanı)** klasörünüze kopyalar (senkronize eder), ardından tüm master havuzu (%80 Train / %20 Valid) oranlarında sıfır kayıpla karıştırıp **`data/`** altına dağıtır ve `data.yaml` dosyasını günceller).*

### 5. Modelleri Eğitme
Modelleri, çalıştırma adımındaki web arayüzü (kontrol paneli) üzerinden doğrudan tarayıcını kullanarak eğitebilirsin.

### 6. Modelleri Karşılaştırma (Metrikler & Grafikler)
Eğittiğin modellerin doğruluk oranlarını ve kayıp (loss) grafiklerini yan yana kıyaslamak için:
```powershell
python scripts/compare_models.py
```
Bu komut kök dizinde **`model_comparison.png`** adında bir grafik oluşturur ve performansları özetler.
