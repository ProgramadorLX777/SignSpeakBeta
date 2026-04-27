import tkinter as tk
import subprocess
import sys
import os

# Ruta base (ajústala si es necesario)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def ejecutar(script):
    ruta = os.path.join(BASE_DIR, script)
    subprocess.Popen([sys.executable, ruta])

# =========================
# INTERFAZ
# =========================
root = tk.Tk()
root.title("SignSpeak Beta")
root.geometry("400x300")

# Tamaño ventana
ancho_ventana = 400
alto_ventana = 400

# Tamaño pantalla
ancho_pantalla = root.winfo_screenwidth()
alto_pantalla = root.winfo_screenheight()

# Calcular posición
x = int((ancho_pantalla / 2) - (ancho_ventana / 2))
y = int((alto_pantalla / 2) - (alto_ventana / 2))

# Aplicar tamaño + posición
root.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

root.configure(bg="#1e1e1e")

titulo = tk.Label(
    root,
    text="SeñasChile(LSCh)",
    font=("Arial", 18, "bold"),
    fg="white",
    bg="#1e1e1e"
)
titulo.pack(pady=20)

# =========================
# BOTONES
# =========================

btn_grabador = tk.Button(
    root,
    text="Grabador Automático",
    width=30,
    height=2,
    command=lambda: ejecutar("grabador_auto_cnn_lstm.py")
)
btn_grabador.pack(pady=5)

btn_entrenador = tk.Button(
    root,
    text="Entrenador Modelo",
    width=30,
    height=2,
    command=lambda: ejecutar("entrenador_cnn_lstm_bimanual.py")
)
btn_entrenador.pack(pady=5)

btn_reconocedor = tk.Button(
    root,
    text="Reconocedor + Traductor",
    width=30,
    height=2,
    command=lambda: ejecutar("reconocedor_traductor.py")
)
btn_reconocedor.pack(pady=5)

btn_visualizador = tk.Button(
    root,
    text="Visualizar Manos",
    width=30,
    height=2,
    command=lambda: ejecutar("visualizacion_manos.py")
)
btn_visualizador.pack(pady=5)

btn_evaluador = tk.Button(
    root,
    text="Evaluador de Modelo",
    width=30,
    height=2,
    command=lambda: ejecutar("evaluacion_modelo.py")
)
btn_evaluador.pack(pady=5)

btn_backup_modelo = tk.Button(
    root,
    text="Crear Backup Modelo",
    width=30,
    height=2,
    command=lambda: ejecutar("backup_modelo.py")
)
btn_backup_modelo.pack(pady=5)

if __name__ == "__main__":
    root.mainloop()