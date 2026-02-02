import cv2
import mediapipe as mp
import numpy as np

# =============================
#        MEDIAPIPE
# =============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# =============================
#          CÁMARA
# =============================
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    print("🟢 Visualización activa")
    print("➡️ Presiona ESC para salir")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Vista espejo (más natural para el usuario)
        frame = cv2.flip(frame, 1)

        # Procesamiento
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # =============================
        #      DIBUJO DE MANOS
        # =============================
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                # Left / Right
                label = results.multi_handedness[idx].classification[0].label

                # Dibujar landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # Coordenada para el texto
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[0].x * w)
                y = int(hand_landmarks.landmark[0].y * h) - 10

                cv2.putText(
                    frame,
                    label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

        # =============================
        #      INFO EN PANTALLA
        # =============================
        cv2.putText(
            frame,
            "Visualizacion de manos - ESC para salir",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.imshow("Visualizacion de Manos", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("⛔ Visualización finalizada")
            break

cap.release()
cv2.destroyAllWindows()
