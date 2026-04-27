import matplotlib.pyplot as plt
import csv

umbrales = [0.6, 0.7, 0.75, 0.8, 0.85]
resultados = []

data = []

with open("log_predicciones.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        real, pred, conf, correcto = row
        data.append((real, pred, float(conf), int(correcto)))

for umbral in umbrales:
    correctos = 0
    total = 0

    for real, pred, conf, correcto in data:
        if conf >= umbral:
            total += 1
            if correcto:
                correctos += 1

    acc = correctos / total if total > 0 else 0
    resultados.append(acc)

plt.plot(umbrales, resultados, marker='o')
plt.title("Precisión vs Umbral (REAL)")
plt.xlabel("Umbral")
plt.ylabel("Precisión")
plt.grid()

plt.savefig("analisis_real.png")
print("📊 Análisis real generado")