# unified_server.py - V12: DOĞRU ORIJINAL MANTIK + 120sn SURE

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
# 1. BÖLÜM: JEST AYARLARI (SENİN ORİJİNAL KODUN)
# ==========================================
ROTATE_FIX = True       # Telefondan gelen görüntüyü düzelt
SAVE_DEBUG = False      # Resim kaydetme YOK

# --- HASSASİYET MANTIĞI ---
PROXIMITY_THRESHOLD = 0.6  
MIN_MOVEMENT = 20          
STABLE_REQUIRED = 0.1      

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def get_face(): return mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
def get_hands(): return mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def calc_dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

MOBILE_SESSIONS = {}

@app.route('/gesture_mobile/start', methods=['POST'])
def start():
    sid = str(uuid4())
    MOBILE_SESSIONS[sid] = {
        "t0": time.time(), 
        "stable_start": None,
        "start_pos": None 
    }
    print(f"📱 Jest Oturumu Başladı: {sid[:5]}")
    return jsonify({"ok": True, "session_id": sid})

@app.route('/gesture_mobile/frame', methods=['POST'])
def frame():
    sid = request.form.get("session_id")
    file = request.files.get("frame")
    
    if sid not in MOBILE_SESSIONS: return jsonify({"detected": False, "message": "Oturum Yok", "final": True})
    st = MOBILE_SESSIONS[sid]

    # --- SADECE BURASI DEĞİŞTİ: 40 YERİNE 120 YAPILDI ---
    if time.time() - st["t0"] > 120:
        del MOBILE_SESSIONS[sid]
        return jsonify({"detected": False, "message": "Zaman Aşımı", "final": True})

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: return jsonify({"detected": False, "message": "Resim Yok", "final": False})
    
    if ROTATE_FIX: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    msg = "Yüz/El Aranıyor..."
    detected = False

    with get_face() as fm, get_hands() as hm:
        f_res = fm.process(rgb)
        h_res = hm.process(rgb)

        if f_res.multi_face_landmarks and h_res.multi_hand_landmarks:
            face = f_res.multi_face_landmarks[0]
            
            # Yüz Referans Noktaları
            forehead = face.landmark[10]
            chin = face.landmark[152]
            face_height = calc_dist((forehead.x*w, forehead.y*h), (chin.x*w, chin.y*h))
            fx, fy = int(forehead.x * w), int(forehead.y * h)

            # TÜM ELLERİ KONTROL ET
            for hand in h_res.multi_hand_landmarks:
                index_tip = hand.landmark[8]
                wrist = hand.landmark[0]
                
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                wx, wy = int(wrist.x * w), int(wrist.y * h)

                is_upright = wy > iy 
                dist_to_forehead = calc_dist((ix, iy), (fx, fy))
                is_close_to_head = dist_to_forehead < (face_height * PROXIMITY_THRESHOLD)

                if st["start_pos"] is None:
                    st["start_pos"] = (ix, iy)
                    st["stable_start"] = time.time()
                    msg = "Hareketi Yap!"
                else:
                    start_ix, start_iy = st["start_pos"]
                    move_dist = calc_dist((ix, iy), (start_ix, start_iy))
                    moved_up = iy < start_iy 

                    msg = f"D:{dist_to_forehead:.0f}"

                    if is_upright and is_close_to_head and move_dist > MIN_MOVEMENT:
                        if moved_up:
                            detected = True
                            msg = "✅ Merhaba Algılandı!"
                            print(f"✅ JEST BAŞARILI! (Alna Yakınlık: {dist_to_forehead:.0f})")
                            del MOBILE_SESSIONS[sid]
                            return jsonify({"detected": True, "message": msg, "final": True})
                        else:
                            msg = "Elini Kaldır"
                    
                    elif not is_close_to_head:
                        msg = "Elini Alnına Getir"
        else:
            msg = "Yüz/El Görülmedi"

    return jsonify({"detected": False, "message": msg, "final": False})

@app.route('/gesture_mobile/end', methods=['POST'])
def end():
    sid = request.form.get("session_id")
    if sid in MOBILE_SESSIONS: del MOBILE_SESSIONS[sid]
    return jsonify({"ok": True})

# ==========================================
# 2. BÖLÜM: SES KISMI (DOKUNULMADI)
# ==========================================
@app.route('/check_speech_audio', methods=['POST'])
def audio():
    # Dosya kontrolü
    if 'file' not in request.files: 
        print("❌ Hata: Ses dosyası gelmedi.")
        return jsonify({"detected": False, "message": "Dosya yok"})
    
    file = request.files['file']
    path = f"temp_{uuid4()}.wav"
    file.save(path)
    
    print(f"🎤 Ses dosyası alındı, işleniyor...")

    r = sr.Recognizer()
    msg = "..."
    detected = False
    
    try:
        with sr.AudioFile(path) as s:
            # Sesi oku
            audio_data = r.record(s)
            
            # Google'a sor
            t = r.recognize_google(audio_data, language="tr-TR").lower()
            print(f"🗣️ Algılanan: {t}")

            # Kelime kontrolü
            if "merhaba" in t or "maraba" in t or "meraba" in t:
                detected = True
                msg = f"✅ {t}"
                print("✅ SES KOMUTU BAŞARILI: Merhaba denildi.")
            else:
                msg = f"Anlaşılan: {t}"
                print(f"❌ Eşleşmedi: {t}")

    except sr.UnknownValueError:
        msg = "Ses anlaşılamadı"
        print("❌ Google sesi anlayamadı.")
    except Exception as e:
        msg = f"Hata: {e}"
        print(f"❌ Hata: {e}")
    
    # Temizlik
    if os.path.exists(path): os.remove(path)
    
    return jsonify({"detected": detected, "message": msg})

if __name__ == '__main__':
    print("🚀 UNIFIED SERVER HAZIR (Jest V8 + Ses)...")
    app.run(host='0.0.0.0', port=5000, threaded=True)