# unified_server.py - V25: ŞAKAK + ALIN DESTEKLİ TİD MERHABA
# Elin şakağa (kaş bitimine) konulmasını da destekler.

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
# AYARLAR
# ==========================================
ROTATE_FIX = True       
# Eşik: Yüz boyunun %60'ı kadar yakınlık yeterli (Çünkü artık tam noktaya bakıyoruz)
PROXIMITY_THRESHOLD = 0.6  
MIN_MOVEMENT = 20          
FINGER_THRESHOLD = 0.08    
MAX_SESSION_TIME = 120      

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def get_face(): return mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
def get_hands(): return mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def calc_dist(p1, p2): return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calc_dist_3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def are_fingers_together(hand_landmarks):
    tips = [8, 12, 16, 20]
    for i in range(len(tips) - 1):
        p1 = hand_landmarks.landmark[tips[i]]
        p2 = hand_landmarks.landmark[tips[i+1]]
        if calc_dist_3d(p1, p2) > FINGER_THRESHOLD:
            return False 
    return True

MOBILE_SESSIONS = {}

@app.route('/gesture_mobile/start', methods=['POST'])
def start():
    sid = str(uuid4())
    MOBILE_SESSIONS[sid] = {
        "t0": time.time(), 
        "start_pos": None,
        "state": "WAITING_HAND" 
    }
    return jsonify({"ok": True, "session_id": sid})

@app.route('/gesture_mobile/frame', methods=['POST'])
def frame():
    try:
        sid = request.form.get("session_id")
        file = request.files.get("frame")
        
        if sid not in MOBILE_SESSIONS: return jsonify({"detected": False, "message": "Oturum Yok", "final": True})
        
        st = MOBILE_SESSIONS[sid]
        if time.time() - st["t0"] > MAX_SESSION_TIME:
            del MOBILE_SESSIONS[sid]
            return jsonify({"detected": False, "message": "Zaman Aşımı", "final": True})

        if not file: return jsonify({"detected": False, "message": "Veri Yok", "final": False})

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None: return jsonify({"detected": False, "message": "Resim Bozuk", "final": False})
        
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
                
                # --- REFERANS NOKTALARI (GÜNCELLENDİ) ---
                # Artık sadece alın ortasına değil, şakaklara da bakıyoruz.
                
                # 1. Alın Ortası (Landmark 10)
                # 2. Sol Kaş Sonu / Şakak (Landmark 334 civarı)
                # 3. Sağ Kaş Sonu / Şakak (Landmark 105 civarı)
                
                ref_points = [
                    face.landmark[10],  # Orta
                    face.landmark[334], # Sol Şakak
                    face.landmark[105]  # Sağ Şakak
                ]
                
                # Yüz boyu referansı (Çene - Alın arası)
                chin = face.landmark[152]
                forehead = face.landmark[10]
                face_height = calc_dist((forehead.x*w, forehead.y*h), (chin.x*w, chin.y*h))

                # EN İYİ ELİ BUL
                active_hand = None
                best_dist = 9999
                
                for hand in h_res.multi_hand_landmarks:
                    if not are_fingers_together(hand): continue 

                    index_tip = hand.landmark[8]
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Elin, HERHANGİ BİR referans noktasına (Alın veya Şakaklar) uzaklığına bak
                    # En yakın olduğu noktayı baz alacağız.
                    min_dist_to_refs = 9999
                    for ref in ref_points:
                        rx, ry = int(ref.x * w), int(ref.y * h)
                        d = calc_dist((ix, iy), (rx, ry))
                        if d < min_dist_to_refs:
                            min_dist_to_refs = d
                    
                    if min_dist_to_refs < best_dist:
                        best_dist = min_dist_to_refs
                        active_hand = hand

                if active_hand:
                    index_tip = active_hand.landmark[8]
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Mesafe Kontrolü (En yakın referans noktasına göre)
                    is_close = best_dist < (face_height * PROXIMITY_THRESHOLD)

                    # --- DURUM MAKİNESİ ---
                    if st["start_pos"] is None:
                        if is_close:
                            st["start_pos"] = (ix, iy) 
                            msg = "Hazır! Selam Ver..."
                        else:
                            msg = "Elini Şakağına/Alnına Koy"
                    
                    else:
                        start_ix, start_iy = st["start_pos"]
                        move_total = calc_dist((ix, iy), (start_ix, start_iy))
                        msg = f"Takipte... M:{move_total:.0f}"

                        # Hareketin Yönü (Opsiyonel Kontrol)
                        # Şakaktan YUKARI veya YANA (Dışarı) olması beklenir.
                        # Aşağı doğru (y artışı) hareketleri eleyebiliriz ama şimdilik esnek bırakıyorum.
                        
                        if move_total > MIN_MOVEMENT:
                            detected_status = True
                            msg = "✅ Merhaba!"
                            final_decision = True
                            del MOBILE_SESSIONS[sid]
                        
                        elif not is_close and move_total > face_height * 1.5:
                             st["start_pos"] = None
                             msg = "Tekrar Dene"
                else:
                    if h_res.multi_hand_landmarks and not active_hand:
                         msg = "Parmaklarını Birleştir"
                    else:
                         msg = "Elini Başına Getir"

            else:
                msg = "Yüz/El Görülmedi"

        return jsonify({"detected": detected_status, "message": msg, "final": final_decision})

    except Exception as e:
        return jsonify({"detected": False, "message": "Sunucu Hatası", "final": True})

@app.route('/gesture_mobile/end', methods=['POST'])
def end():
    sid = request.form.get("session_id")
    if sid in MOBILE_SESSIONS: del MOBILE_SESSIONS[sid]
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
                msg = f"✅ Algılandı: {t}"
            else:
                msg = f"Farklı: {t}"
    except:
        msg = "Anlaşılamadı"
    if os.path.exists(path): os.remove(path)
    return jsonify({"detected": detected, "message": msg})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
