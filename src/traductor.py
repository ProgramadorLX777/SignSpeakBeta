import cv2
import numpy as np

class Traductor:
    def __init__(self, bus):
        self.texto = ""
        self.confianza = 0.0

        bus.suscribir("SENIA_DETECTADA", self.recibir)

    def recibir(self, datos):
        self.texto = datos["label"]
        self.confianza = datos["confianza"]

    def dibujar(self):
        if not self.texto:
            return

        # Fondo blanco
        frame = np.ones((200, 600, 3), dtype=np.uint8) * 255

        cv2.putText(
            frame,
            f"TEXTO: {self.texto}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 150, 0),
            3,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Confianza: {self.confianza:.2f}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (80, 80, 80),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("📝 Traductor", frame)
