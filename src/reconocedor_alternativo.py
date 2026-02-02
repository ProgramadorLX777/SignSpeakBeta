import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import joblib
from collections import deque
import time

# =========================
# CONFIGURACIÓN
# =========================
MODEL_PATH = "models/modelo_cnn_lstm_bimanual.pth"
LABELS_PATH = "models/labels_bimano.pkl"
SEQ_LEN = 50
MIN_FRAMES = 15
FEATURES = 126
UMBRAL_CONF = 0.70
TIEMPO_MOSTRAR = 2.0  # segundos

# =========================
# MENSAJES UX POR COLOR
# =========================
MENSAJE_ROJO = "= Confianza baja"
MENSAJE_AMARILLO = "= Casi listo..."
MENSAJE_VERDE = "= Excelente"

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# MODELO
# =========================
class CNN1D_LSTM(nn.Module):
    def __init__(self, features, hidden_lstm=128, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv1d(features, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            128,
            hidden_lstm,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(hidden_lstm * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)

# =========================
# CARGA MODELO
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
id_to_label = joblib.load(LABELS_PATH)

model = CNN1D_LSTM(FEATURES, 128, len(id_to_label)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("🔥 Preparando modelo...")

dummy_seq = torch.zeros((1, SEQ_LEN, FEATURES), dtype=torch.float32).to(device)

with torch.no_grad():
    for _ in range(3):
        _ = model(dummy_seq)

print("✅ Modelo listo")

# =========================
# FUNCIONES
# =========================
def extraer_landmarks(result):
    mano_left = np.zeros((21, 3))
    mano_right = np.zeros((21, 3))

    if not result.multi_hand_landmarks:
        return np.concatenate([mano_left.flatten(), mano_right.flatten()]), False

    if result.multi_handedness:
        for lm, hand_h in zip(result.multi_hand_landmarks, result.multi_handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
            if hand_h.classification[0].label == "Left":
                mano_left = pts
            else:
                mano_right = pts

    manos_detectadas = not (
        np.allclose(mano_left, 0) and np.allclose(mano_right, 0)
    )

    return np.concatenate([mano_left.flatten(), mano_right.flatten()]), manos_detectadas

# =========================
# RECONOCIMIENTO
# =========================
ultimo_resultado = ""
ultimo_tiempo = 0.0
sin_manos_frames = 0
bloqueado = False
tiempo_bloqueo = 0.0
MAX_SIN_MANOS = 10
TIEMPO_BLOQUEO = 1.5

cap = cv2.VideoCapture(0)
window = deque(maxlen=SEQ_LEN)

print("🎥 Reconocedor activado (Presione ESC para salir!!)")

color = (255, 255, 255)  # blanco por defecto
mensaje_estado = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    # =========================
    # DIBUJAR LANDMARKS
    # =========================
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    #color=(0, 255, 0),
                    #color=(0, 255, 255),
                    # LOS COLORES ESTAN BGR NO EN RGB
                    color=(0, 0, 255),
                    thickness=2,
                    circle_radius=3
                ),
                mp_drawing.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=2
                )
            )

    vec, manos = extraer_landmarks(result)

    # Si no hay manos → no acumulamos
    if not manos:
        sin_manos_frames += 1
        if sin_manos_frames > MAX_SIN_MANOS:
            pass
           #window.clear()
    else:
        sin_manos_frames = 0
        window.append(vec)

    # CUANDO SE COMPLETA LA SECUENCIA
    detectando = False
    
    if len(window) >= MIN_FRAMES:
        detectando = True
        print("\n🧠 MODELO EJECUTÁNDOSE")

        seq_raw = np.array(window, dtype=np.float32)

        # Padding si hay menos frames que SEQ_LEN
        if len(seq_raw) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(seq_raw), FEATURES), dtype=np.float32)
            seq = np.vstack([pad, seq_raw])
        else:
            seq = seq_raw[-SEQ_LEN:]

        # Normalización
        seq = (seq - seq.mean()) / (seq.std() + 1e-6)
        tensor = torch.tensor(seq).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()

        for i, p in enumerate(probs):
            print(f"  {id_to_label[i]}: {p:.3f}")

        pred_id = int(np.argmax(probs))
        conf = probs[pred_id]

        print("➡ Prediccion:", id_to_label[pred_id], "Confianza:", conf)

        if not bloqueado and conf >= UMBRAL_CONF:
            ultimo_resultado = id_to_label[pred_id]
            ultimo_tiempo = time.time()
            tiempo_bloqueo = time.time()
            bloqueado = True
            window.clear()
            
        if bloqueado and sin_manos_frames > MAX_SIN_MANOS:
            bloqueado = False
            
        elif bloqueado and (time.time() - tiempo_bloqueo > TIEMPO_BLOQUEO):
            bloqueado = False
            
            # MOSTRAR TEXTO EN PANTALLA
            if conf >= 0.80:
                color = (0, 255, 0)      # verde
                mensaje_estado = MENSAJE_VERDE
                
            elif conf >= 0.70:
                color = (0, 255, 255)    # amarillo
                mensaje_estado = MENSAJE_AMARILLO
                
            else:
                color = (0, 0, 255)      # rojo
                mensaje_estado = MENSAJE_ROJO
                
    if ultimo_resultado and (time.time() - ultimo_tiempo <= TIEMPO_MOSTRAR):
        
        cv2.putText(
            frame,
            ultimo_resultado,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),   # negro
            10,         # grosor contorno
            lineType=cv2.LINE_AA
        )
        
        cv2.putText(
            frame,
            ultimo_resultado,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            #(0, 255, 0),
            3,
            lineType=cv2.LINE_AA
        )
        
        # ===== CALCULAR POSICIÓN A LA DERECHA =====
        (text_w, text_h), _ = cv2.getTextSize(
            ultimo_resultado,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            3
        )

        x_mensaje = 10 + text_w + 15
        y_mensaje = 40

        # ===== MENSAJE DE ESTADO (NO REEMPLAZA LA SEÑA) =====
        cv2.putText(
            frame,
            mensaje_estado,
            (x_mensaje, y_mensaje),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            6,
            lineType=cv2.LINE_AA
        )

        cv2.putText(
            frame,
            mensaje_estado,
            (x_mensaje, y_mensaje),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            lineType=cv2.LINE_AA
        )
        
    elif detectando and not bloqueado:
        '''# texto = "Detectando..."
        # color_det = (255, 200, 0)  # amarillo suave
        color_det = (255, 255, 0)  # amarillo suave

        cv2.putText(
            frame,
            texto,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            8,
            lineType=cv2.LINE_AA
        )

        cv2.putText(
            frame,
            texto,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color_det,
            2,
            lineType=cv2.LINE_AA
        )'''
        print("AZUL")
        
    else:
        texto = "Esperando Senia..."
        color_espera = (200, 200, 200)  # gris claro

        # CONTORNO NEGRO
        cv2.putText(
            frame,
            texto,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            10,
            lineType=cv2.LINE_AA
        )

        # TEXTO PRINCIPAL
        cv2.putText(
            frame,
            texto,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color_espera,
            2,
            lineType=cv2.LINE_AA
        )    

    cv2.imshow("Reconocedor Activo - Presione ESC para Salir!!", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()