# -*- coding: utf-8 -*-
import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("      VOLLEYBALL DETECTION - LABELIMG LAUNCHER")
    print("=" * 60)
    
    # 1. PyQt5 eklenti yolunu dinamik bulalım
    try:
        import PyQt5
        pyqt_dir = os.path.dirname(PyQt5.__file__)
        plugins_dir = os.path.join(pyqt_dir, "Qt5", "plugins")
        if os.path.exists(plugins_dir):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins_dir
            print(f"[OK] Qt platform plugin path ayarlandi: {plugins_dir}")
        else:
            print("[UYARI] Qt5 plugins dizini bulunamadi.")
    except ImportError:
        print("[HATA] PyQt5 kütüphanesi sanal ortamda bulunamadi!")
        return

    # 2. Çalıştırılacak script'in yolunu bulalım
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(project_dir, ".venv", "Scripts", "labelImg-script.py")
    
    if not os.path.exists(script_path):
        print(f"[HATA] labelImg-script.py bulunamadi: {script_path}")
        return

    print("[INFO] labelImg arka planda baslatiliyor...")
    
    # 3. Bağımsız (detached) bir konsolda çalıştıralım
    # Bu sayede terminaliniz açık kalır ve etiketleme ekranı bağımsız açılır.
    args = [
        sys.executable, 
        script_path, 
        "manual_labeling/images", 
        "manual_labeling/images/classes.txt", 
        "manual_labeling/labels"
    ]
    
    try:
        subprocess.Popen(
            args,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            close_fds=True
        )
        print("[BAŞARILI] labelImg başarıyla açıldı. İyi etiketlemeler!")
    except Exception as e:
        print(f"[HATA] labelImg başlatılamadı: {e}")
        
    print("=" * 60)

if __name__ == '__main__':
    main()
