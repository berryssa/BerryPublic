# unified_server.py - V14 FINAL: KUSURSUZ JEST + SES + DONMA ÖNLEYİCİ

from flask import Flask, jsonify, request
import cv2
import mediapipe as mp
import math
import time
import numpy as np
from uuid import uuid4
import os
import speech_recognition as sr

app = Flask(__name__)

# ==========================================
# AYARLAR (HASSASİYET VE PERFORMANS)
# ==========================================
ROTATE_FIX = True       # Telefondan gelen dikey görüntüyü düzelt
PROXIMITY_THRESHOLD = 0.85  # El, yüz boyutunun %85'i kadar yakında olabilir (Şakak/Kaş üstü dahil)
MIN_MOVEMENT = 15           # Hareket algılama eşiği (Daha hassas)
MAX_SESSION_TIME = 120      # 120 saniye sonra oturumu zorla kapat (Hafıza şişmesini önler)

# MediaPipe Kurulumu
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def get_face(): return mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
def get_hands(): return mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def calc_dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Oturumları Hafızada Tut
MOBILE_SESSIONS = {}

# ==========================================
# 1. MODÜL: JEST ALGILAMA (GÖRÜNTÜ)
# ==========================================

@app.route('/gesture_mobile/start', methods=['POST'])
def start():
    # Yeni bir oturum başlatır
    sid = str(uuid4())
    MOBILE_SESSIONS[sid] = {
        "t0": time.time(), 
        "start_pos": None 
    }
    print(f"📱 [JEST] Oturum Başladı: {sid[:5]}")
    return jsonify({"ok": True, "session_id": sid})

@app.route('/gesture_mobile/frame', methods=['POST'])
def frame():
    try:
        sid = request.form.get("session_id")
        file = request.files.get("frame")
        
        # 1. Kontrol: Oturum var mı?
        if sid not in MOBILE_SESSIONS: 
            return jsonify({"detected": False, "message": "Oturum Yok/Bitti", "final": True})
        
        st = MOBILE_SESSIONS[sid]

        # 2. Kontrol: Zaman Aşımı (Donmayı engeller)
        if time.time() - st["t0"] > MAX_SESSION_TIME:
            del MOBILE_SESSIONS[sid]
            return jsonify({"detected": False, "message": "Zaman Aşımı", "final": True})

        # 3. Kontrol: Resim okuma
        if not file:
            return jsonify({"detected": False, "message": "Veri Yok", "final": False})

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None: 
            return jsonify({"detected": False, "message": "Resim Bozuk", "final": False})
        
        if ROTATE_FIX: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        msg = "Yüz/El Aranıyor..."
        final_decision = False
        detected_status = False

        with get_face() as fm, get_hands() as hm:
            f_res = fm.process(rgb)
            h_res = hm.process(rgb)

            if f_res.multi_face_landmarks and h_res.multi_hand_landmarks:
                face = f_res.multi_face_landmarks[0]
                
                # Yüz Referansları
                forehead = face.landmark[10]
                chin = face.landmark[152]
                face_height = calc_dist((forehead.x*w, forehead.y*h), (chin.x*w, chin.y*h))
                fx, fy = int(forehead.x * w), int(forehead.y * h)

                # Elleri Kontrol Et (Herhangi biri uyarsa yeterli)
                hand_found_near_head = False
                
                for i, hand in enumerate(h_res.multi_hand_landmarks):
                    index_tip = hand.landmark[8]
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Mesafe Kontrolü
                    dist_to_forehead = calc_dist((ix, iy), (fx, fy))
                    is_close = dist_to_forehead < (face_height * PROXIMITY_THRESHOLD)

                    if is_close: hand_found_near_head = True

                    # --- HAREKET MANTIĞI ---
                    if st["start_pos"] is None:
                        # Henüz başlangıç yapılmadı, el kafaya yakınsa kaydet
                        if is_close:
                            st["start_pos"] = (ix, iy)
                            msg = "Hazır! Hareketi Yap..."
                        else:
                            msg = "Elini Başına Getir"
                    else:
                        # Hareket başlamış, analiz et
                        start_ix, start_iy = st["start_pos"]
                        move_total = calc_dist((ix, iy), (start_ix, start_iy))
                        
                        diff_y = iy - start_iy  # Negatif=Yukarı
                        diff_x = abs(ix - start_ix) # Yana açılma

                        msg = f"Takipte... Hareket: {move_total:.0f}"

                        # Eğer hareket yeterince büyükse
                        if move_total > MIN_MOVEMENT:
                            # Yukarı hareket VEYA Yana Hareket
                            is_moving_up = diff_y < 0
                            is_moving_side = diff_x > (MIN_MOVEMENT * 0.8)

                            if is_moving_up or is_moving_side:
                                print(f"✅ [JEST] MERHABA ALGILANDI! (Mesafe: {move_total:.1f})")
                                detected_status = True
                                msg = "✅ Merhaba!"
                                final_decision = True # Döngüyü kır ve bitir
                                del MOBILE_SESSIONS[sid] # Temizle
                                break # For döngüsünden çık
                        
                        # El kafadan uzaklaştı ama hareket algılanmadıysa (Hata toleransı)
                        elif not is_close and move_total > face_height:
                            st["start_pos"] = None
                            msg = "Tekrar Dene"

            else:
                msg = "Yüz/El Görülmedi"

        return jsonify({"detected": detected_status, "message": msg, "final": final_decision})

    except Exception as e:
        print(f"❌ [HATA] Frame Hatası: {e}")
        return jsonify({"detected": False, "message": "Sunucu Hatası", "final": True})

@app.route('/gesture_mobile/end', methods=['POST'])
def end():
    # Unity tarafı oturumu manuel bitirmek isterse
    sid = request.form.get("session_id")
    if sid in MOBILE_SESSIONS: del MOBILE_SESSIONS[sid]
    return jsonify({"ok": True})


# ==========================================
# 2. MODÜL: SES TANIMA (İŞARET DİLİ BUTONU İÇİN)
# ==========================================

@app.route('/check_speech_audio', methods=['POST'])
def audio():
    # Bu fonksiyon "İşaret Dili" butonunun çalışmasını sağlar.
    if 'file' not in request.files: 
        return jsonify({"detected": False, "message": "Dosya yok"})
    
    file = request.files['file']
    path = f"temp_{uuid4()}.wav"
    file.save(path)
    
    r = sr.Recognizer()
    msg = "Ses Anlaşılamadı"
    detected = False
    
    try:
        print("🎤 [SES] Dosya işleniyor...")
        with sr.AudioFile(path) as s:
            audio_data = r.record(s)
            # Google ses tanıma servisine gönder
            t = r.recognize_google(audio_data, language="tr-TR").lower()
            print(f"🗣️ [SES] Algılanan: {t}")
            
            # Kelime Kontrolü
            if "merhaba" in t or "maraba" in t or "meraba" in t or "selam" in t:
                detected = True
                msg = f"✅ Algılandı: {t}"
            else:
                msg = f"Farklı Kelime: {t}"
                
    except sr.UnknownValueError:
        msg = "Ses Anlaşılamadı (Gürültü?)"
        print("❌ [SES] Google anlayamadı.")
    except Exception as e:
        msg = f"Hata: {str(e)}"
        print(f"❌ [SES] Hata: {e}")
    
    # Geçici dosyayı temizle (Hafıza dolmasın)
    if os.path.exists(path): os.remove(path)
    
    return jsonify({"detected": detected, "message": msg})

if __name__ == '__main__':
    print("🚀 UNIFIED SERVER V14 FINAL HAZIR...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
