import mediapipe as mp
import cv2
from others.dibujo_landmarks import draw_hand_landmarks

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def draw_hand_landmarks(
    frame,
    result,
    landmark_color=(255, 0, 0),        # 🔵 puntos (BGR)
    connection_color=(255, 255, 255),  # ⚪ líneas
    landmark_thickness=2,
    connection_thickness=2,
    landmark_radius=3
):
    """
    Dibuja landmarks de manos sobre el frame.
    """

    if not result.multi_hand_landmarks:
        return frame

    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=landmark_color,
                thickness=landmark_thickness,
                circle_radius=landmark_radius
            ),
            mp_drawing.DrawingSpec(
                color=connection_color,
                thickness=connection_thickness
            )
        )

    return frame
