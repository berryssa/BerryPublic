# unified_server.py - V27: RAM OPTİMİZASYONLU (ÇÖKMEYEN) VERSİYON
# MediaPipe modellerini global yaparak RAM kullanımını sabitler.

from flask import Flask, jsonify, request
import cv2
import mediapipe as mp
import math
import time
import numpy as np
from uuid import uuid4
import os
import speech_recognition as sr
import gc # Çöp toplayıcı (RAM temizliği için)

app = Flask(__name__)

# ==========================================
# AYARLAR
# ==========================================
ROTATE_FIX = True       
PROXIMITY_THRESHOLD = 0.6  
MIN_MOVEMENT = 15          
FINGER_THRESHOLD = 0.08    
MAX_SESSION_TIME = 120      

# ==========================================
# 🧠 YAPAY ZEKA MODELLERİ (GLOBAL)
# Eskiden her karede baştan yaratılıyordu, şimdi 1 kere yaratılıp hep kullanılacak.
# ==========================================
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# RAM TASARRUFU: refine_landmarks=False (Göz bebeği takibi yok)
face_mesh = mp_face.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=False, 
    min_detection_confidence=0.5
)

# RAM TASARRUFU: max_num_hands=1 (Tek el yeterli ve hızlı)
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

def calc_dist(p1, p2): return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
def calc_dist_3d(p1, p2): return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def are_fingers_together(hand_landmarks):
    tips = [8, 12, 16, 20]
    for i in range(len(tips) - 1):
        p1 = hand_landmarks.landmark[tips[i]]
        p2 = hand_landmarks.landmark[tips[i+1]]
        if calc_dist_3d(p1, p2) > FINGER_THRESHOLD: return False 
    return True

MOBILE_SESSIONS = {}

@app.route('/gesture_mobile/start', methods=['POST'])
def start():
    # RAM Temizliği: Yeni oturum başlarken eski çöpleri at
    gc.collect()
    
    sid = str(uuid4())
    MOBILE_SESSIONS[sid] = { "t0": time.time(), "start_pos": None, "state": "WAITING_HAND" }
    return jsonify({"ok": True, "session_id": sid})

@app.route('/gesture_mobile/frame', methods=['POST'])
def frame():
    try:
        sid = request.form.get("session_id")
        file = request.files.get("frame")
        
        if sid not in MOBILE_SESSIONS: return jsonify({"detected": False, "message": "Oturum Yok", "final": True})
        
        st = MOBILE_SESSIONS[sid]
        # Zaman aşımı kontrolü
        if time.time() - st["t0"] > MAX_SESSION_TIME:
            del MOBILE_SESSIONS[sid]
            return jsonify({"detected": False, "message": "Zaman Aşımı", "final": True})

        if not file: return jsonify({"detected": False, "message": "Veri Yok", "final": False})

        # Resmi oku
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None: return jsonify({"detected": False, "message": "Resim Bozuk", "final": False})
        
        if ROTATE_FIX: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        msg = "Yüz/El Aranıyor..."
        final_decision = False
        detected_status = False

        # GLOBAL MODELLERİ KULLAN (with bloğu yok artık)
        f_res = face_mesh.process(rgb)
        h_res = hands.process(rgb)

        if f_res.multi_face_landmarks and h_res.multi_hand_landmarks:
            face = f_res.multi_face_landmarks[0]
            
            # Referanslar
            ref_points = [face.landmark[10], face.landmark[334], face.landmark[105]]
            
            chin = face.landmark[152]
            forehead = face.landmark[10]
            face_height = calc_dist((forehead.x*w, forehead.y*h), (chin.x*w, chin.y*h))

            # Tek el modu (RAM için)
            hand = h_res.multi_hand_landmarks[0]
            
            if are_fingers_together(hand):
                index_tip = hand.landmark[8]
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                
                # Mesafeleri ölç
                min_dist_to_refs = 9999
                for ref in ref_points:
                    rx, ry = int(ref.x * w), int(ref.y * h)
                    d = calc_dist((ix, iy), (rx, ry))
                    if d < min_dist_to_refs: min_dist_to_refs = d
                
                is_close = min_dist_to_refs < (face_height * PROXIMITY_THRESHOLD)

                if st["start_pos"] is None:
                    if is_close:
                        st["start_pos"] = (ix, iy) 
                        msg = "Hazır! Selam Ver..."
                    else:
                        msg = "Elini Başına Getir"
                else:
                    start_ix, start_iy = st["start_pos"]
                    move_total = calc_dist((ix, iy), (start_ix, start_iy))
                    msg = f"Takipte... M:{move_total:.0f}"

                    if move_total > MIN_MOVEMENT:
                        detected_status = True
                        msg = "✅ Merhaba!"
                        final_decision = True
                        del MOBILE_SESSIONS[sid]
                    elif not is_close and move_total > face_height * 1.5:
                            st["start_pos"] = None
                            msg = "Tekrar Dene"
            else:
                msg = "Parmaklarını Birleştir"

        else:
            msg = "Yüz/El Görülmedi" if not f_res.multi_face_landmarks else "El Görülmedi"

        return jsonify({"detected": detected_status, "message": msg, "final": final_decision})

    except Exception as e:
        print(f"Hata: {e}")
        return jsonify({"detected": False, "message": "Sunucu Hatası", "final": True})

@app.route('/gesture_mobile/end', methods=['POST'])
def end():
    sid = request.form.get("session_id")
    if sid in MOBILE_SESSIONS: del MOBILE_SESSIONS[sid]
    gc.collect() # RAM Temizliği
    return jsonify({"ok": True})

@app.route('/check_speech_audio', methods=['POST'])
def audio():
    if 'file' not in request.files: return jsonify({"detected": False, "message": "Dosya yok"})
    file = request.files['file']
    path = f"temp_{uuid4()}.wav"
    file.save(path)
    r = sr.Recognizer()
    msg = "Ses Anlaşılamadı"
    detected = False
    try:
        with sr.AudioFile(path) as s:
            audio_data = r.record(s)
            t = r.recognize_google(audio_data, language="tr-TR").lower()
            if "merhaba" in t or "maraba" in t or "selam" in t:
                detected = True
                msg = f"✅ {t}"
            else:
                msg = f"{t}"
    except:
        msg = "Anlaşılamadı"
    if os.path.exists(path): os.remove(path)
    gc.collect() # RAM Temizliği
    return jsonify({"detected": detected, "message": msg})

if __name__ == '__main__':
    # Threaded=False yaparak RAM kullanımını daha da düşürebiliriz gerekirse
    app.run(host='0.0.0.0', port=5000)
