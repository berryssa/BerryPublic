# unified_server.py - V13: ESNEK VE KUSURSUZ MERHABA (TID)

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
# 1. BÖLÜM: JEST AYARLARI 
# ==========================================
ROTATE_FIX = True       # Telefondan gelen görüntüyü düzelt
SAVE_DEBUG = False      # Resim kaydetme

# --- YENİ HASSASİYET MANTIĞI ---
# 0.6 -> 0.85: Elin, yüz yüksekliğinin %85'i kadar bir çapta (şakaklar dahil) olması yeterli.
PROXIMITY_THRESHOLD = 0.85  
# 20 -> 15: Daha ufak ve seri hareketleri algılaması için düşürüldü.
MIN_MOVEMENT = 15           

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

    # --- ZAMAN AŞIMI (120 sn) ---
    if time.time() - st["t0"] > 120:
        del MOBILE_SESSIONS[sid]
        return jsonify({"detected": False, "message": "Zaman Aşımı", "final": True})

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: return jsonify({"detected": False, "message": "Resim Yok", "final": False})
    
    if ROTATE_FIX: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    msg = "Yüz/El Aranıyor..."
    
    with get_face() as fm, get_hands() as hm:
        f_res = fm.process(rgb)
        h_res = hm.process(rgb)

        if f_res.multi_face_landmarks and h_res.multi_hand_landmarks:
            face = f_res.multi_face_landmarks[0]
            
            # Yüz Referans Noktaları (Alın ve Çene)
            forehead = face.landmark[10]
            chin = face.landmark[152]
            face_height = calc_dist((forehead.x*w, forehead.y*h), (chin.x*w, chin.y*h))
            fx, fy = int(forehead.x * w), int(forehead.y * h)

            # TÜM ELLERİ KONTROL ET (Sağ veya Sol fark etmez)
            for i, hand in enumerate(h_res.multi_hand_landmarks):
                
                index_tip = hand.landmark[8]  # İşaret parmağı ucu
                
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                
                # El alına ne kadar yakın?
                dist_to_forehead = calc_dist((ix, iy), (fx, fy))
                
                # Eşik Kontrolü: Yüz boyunun %85'i kadar yakın mı? (Şakaklar, kaş üstü vs.)
                is_close_to_head = dist_to_forehead < (face_height * PROXIMITY_THRESHOLD)

                # --- SENARYO 1: HAREKET BAŞLANGICI ---
                if st["start_pos"] is None:
                    if is_close_to_head:
                        st["start_pos"] = (ix, iy)
                        msg = "Hazır! Hareketi Yap..."
                        # Başlangıç anında elin konumunu kaydettik
                    else:
                        msg = "Elini Alnına/Başına Getir"
                
                # --- SENARYO 2: HAREKET ANALİZİ ---
                else:
                    start_ix, start_iy = st["start_pos"]
                    
                    # Ne kadar hareket etti?
                    move_total = calc_dist((ix, iy), (start_ix, start_iy))
                    
                    # Yön Analizi
                    diff_y = iy - start_iy  # Negatifse Yukarı, Pozitifse Aşağı
                    diff_x = abs(ix - start_ix) # Yana açılma miktarı
                    
                    msg = f"Mesafe: {dist_to_forehead:.0f} Hareket: {move_total:.0f}"

                    # KURAL: El önceden kafadaydı (start_pos doldu).
                    # Şimdi el hareket ettiyse (MIN_MOVEMENT kadar)...
                    if move_total > MIN_MOVEMENT:
                        
                        # Hareket Yönü Kontrolü:
                        # 1. Yukarı çıkıyor mu? (diff_y < 0)
                        # 2. VEYA Yana doğru (sağa/sola) açılıyor mu? (diff_x > MIN_MOVEMENT)
                        # KISITLAMA: Sadece aşağı inmesini (diff_y > 0) istemiyoruz, bu "el indirme" olur.
                        # Ama TİD Merhaba'da el yana doğru açılırken hafif aşağı da inebilir, o yüzden 
                        # sadece "kafa hizasından uzaklaşma" mantığına bakıyoruz.
                        
                        is_moving_up = diff_y < 0
                        is_moving_side = diff_x > (MIN_MOVEMENT * 0.8) # Yana hareket

                        if is_moving_up or is_moving_side:
                            print(f"✅ MERHABA ALGILANDI! (Hareket: {move_total:.1f})")
                            del MOBILE_SESSIONS[sid]
                            return jsonify({"detected": True, "message": "✅ Merhaba!", "final": True})
                        else:
                            msg = "Hareketi Tamamla (Yukarı/Yana)"
                    
                    # Eğer el kafadan çok uzaklaştıysa ve hareket algılanmadıysa sıfırla
                    elif not is_close_to_head and move_total > face_height:
                         st["start_pos"] = None
                         msg = "Tekrar Dene"

            # Döngü bitti, eğer return olmadıysa:
            return jsonify({"detected": False, "message": msg, "final": False})

        else:
            return jsonify({"detected": False, "message": "Yüz/El Görülmedi", "final": False})

# ==========================================
# 2. BÖLÜM: SES KISMI (DOKUNULMADI)
# ==========================================
@app.route('/check_speech_audio', methods=['POST'])
def audio():
    if 'file' not in request.files: return jsonify({"detected": False, "message": "Dosya yok"})
    file = request.files['file']
    path = f"temp_{uuid4()}.wav"
    file.save(path)
    r = sr.Recognizer()
    msg = "..."
    detected = False
    try:
        with sr.AudioFile(path) as s:
            audio_data = r.record(s)
            t = r.recognize_google(audio_data, language="tr-TR").lower()
            print(f"🗣️ Algılanan: {t}")
            if "merhaba" in t or "maraba" in t or "meraba" in t:
                detected = True
                msg = f"✅ {t}"
            else:
                msg = f"Anlaşılan: {t}"
    except Exception as e:
        msg = "Ses anlaşılamadı"
    if os.path.exists(path): os.remove(path)
    return jsonify({"detected": detected, "message": msg})

if __name__ == '__main__':
    print("🚀 UNIFIED SERVER V13 HAZIR...")
    app.run(host='0.0.0.0', port=5000, threaded=True)
