# bus_eventos.py

class BusEventos:
    def __init__(self):
        self.suscriptores = {}

    def suscribir(self, evento, callback):
        if evento not in self.suscriptores:
            self.suscriptores[evento] = []

        self.suscriptores[evento].append(callback)

    def publicar(self, evento, datos):
        if evento in self.suscriptores:
            for callback in self.suscriptores[evento]:
                callback(datos)
