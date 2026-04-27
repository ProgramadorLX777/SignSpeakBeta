import os
import shutil
from datetime import datetime

def crear_backup_completo(ruta_modelo, ruta_labels):
    # verificar existencia
    if not os.path.exists(ruta_modelo):
        print("❌ Modelo no encontrado")
        return
    
    if not os.path.exists(ruta_labels):
        print("❌ Labels no encontrados")
        return

    # carpeta backups
    carpeta_base = os.path.dirname(ruta_modelo)
    carpeta_backup = os.path.join(carpeta_base, "backups")
    os.makedirs(carpeta_backup, exist_ok=True)

    # timestamp único
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # nombres
    nombre_modelo = f"modelo_{timestamp}.pth"
    nombre_labels = f"labels_{timestamp}.pkl"

    destino_modelo = os.path.join(carpeta_backup, nombre_modelo)
    destino_labels = os.path.join(carpeta_backup, nombre_labels)

    # copiar archivos
    shutil.copy2(ruta_modelo, destino_modelo)
    shutil.copy2(ruta_labels, destino_labels)

    print("✅ Backup completo creado:")
    print("   Modelo:", destino_modelo)
    print("   Labels:", destino_labels)
    
if __name__ == "__main__":
    crear_backup_completo(
        "models/modelo_cnn_lstm_bimanual.pth",
        "models/labels_bimano.pkl"
    )