# unified_server.py - V24: HER İKİ EL, GENİŞ ALAN, TİD UYUMLU
# Sağ/Sol el fark etmez, kafa üstü/şakak fark etmez.

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
# AYARLAR (SENİN İSTEKLERİNE GÖRE GÜNCELLENDİ)
# ==========================================
ROTATE_FIX = True       
# Eşik Arttırıldı: 1.0 = Yüzün boyu kadar mesafe. 
# Bu, elini kafanın tepesine de koysan, şakağa da koysan algılamasını sağlar.
PROXIMITY_THRESHOLD = 1.0  
MIN_MOVEMENT = 20          
FINGER_THRESHOLD = 0.08    
MAX_SESSION_TIME = 120      

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def get_face(): return mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
# GÜNCELLEME: max_num_hands=2 yapıldı. İki el de ekrandaysa ikisine de bakar.
def get_hands(): return mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def calc_dist(p1, p2): return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calc_dist_3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# PARMAK BİTİŞİK Mİ?
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
                
                # Yüz Referansları
                forehead = face.landmark[10]
                chin = face.landmark[152]
                face_height = calc_dist((forehead.x*w, forehead.y*h), (chin.x*w, chin.y*h))
                fx, fy = int(forehead.x * w), int(forehead.y * h)

                # --- ÇOKLU EL DESTEĞİ ---
                # Ekrandaki tüm ellere bak, hangisi "aktif" ise onu kullan.
                active_hand = None
                
                # Önce alına en yakın ve parmakları bitişik olan eli bulalım
                best_dist = 9999
                
                for hand in h_res.multi_hand_landmarks:
                    # 1. Parmak kontrolü (Tüm eller için şart)
                    if not are_fingers_together(hand): continue 

                    index_tip = hand.landmark[8]
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    dist = calc_dist((ix, iy), (fx, fy))
                    
                    # Eğer bu el alına, diğerinden daha yakınsa bunu seç
                    if dist < best_dist:
                        best_dist = dist
                        active_hand = hand

                # Eğer uygun bir el bulunduysa işlemlere devam et
                if active_hand:
                    index_tip = active_hand.landmark[8]
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Mesafe Kontrolü (Genişletilmiş Alan: Kafa üstü dahil)
                    is_close = best_dist < (face_height * PROXIMITY_THRESHOLD)

                    # --- DURUM MAKİNESİ ---
                    
                    # DURUM 1: Başlangıç
                    if st["start_pos"] is None:
                        if is_close:
                            st["start_pos"] = (ix, iy) 
                            msg = "Hazır! Selam Ver..."
                        else:
                            # Eğer uygun el var ama uzaktaysa
                            msg = "Elini Başına Getir"
                    
                    # DURUM 2: Hareket Takibi
                    else:
                        start_ix, start_iy = st["start_pos"]
                        move_total = calc_dist((ix, iy), (start_ix, start_iy))
                        msg = f"Takipte... M:{move_total:.0f}"

                        if move_total > MIN_MOVEMENT:
                            # HAREKET BAŞARILI (Yön fark etmeksizin uzaklaşma)
                            detected_status = True
                            msg = "✅ Merhaba!"
                            final_decision = True
                            del MOBILE_SESSIONS[sid]
                        
                        # Hata durumu: El çok uzaklaştı ama hareket sayılmadı
                        elif not is_close and move_total > face_height * 1.5:
                             st["start_pos"] = None
                             msg = "Tekrar Dene"
                else:
                    # El var ama parmaklar bitişik değilse veya çok uzaktaysa
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
