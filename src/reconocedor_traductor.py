from bus_eventos import BusEventos
from traductor import Traductor
import reconocedor_alternativo

bus = BusEventos()
traductor = Traductor(bus)

reconocedor_alternativo.iniciar(bus, traductor)
