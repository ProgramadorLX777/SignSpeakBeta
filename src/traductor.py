import cv2
import numpy as np

class Traductor:
    def __init__(self, bus):
        self.texto = ""
        self.confianza = 0.0

        self.ultimo_tiempo = 0.0
        TIEMPO_LIMPIAR = 2.0
        self.tiempo_limpiar = TIEMPO_LIMPIAR

        bus.suscribir("SENIA_DETECTADA", self.recibir)

    def recibir(self, datos):
        nuevo_texto = datos["label"]

        # SOLO si cambia la seña
        if nuevo_texto != self.texto:
            self.texto = nuevo_texto
            self.confianza = datos["confianza"]
            self.ultimo_tiempo = cv2.getTickCount() / cv2.getTickFrequency()

    def dibujar(self):
        tiempo_actual = cv2.getTickCount() / cv2.getTickFrequency()

        # limpiar si pasa el tiempo
        if self.texto and (tiempo_actual - self.ultimo_tiempo > self.tiempo_limpiar):
            self.texto = ""
            self.confianza = 0.0
            return

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
