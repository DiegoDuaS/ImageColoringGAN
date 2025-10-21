import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Directorio base
INPUT_DIR = "ImagesRaw"

# Datos recolectados
data = []

print("Analizando dataset...")

# Recorre cada clase
for category in os.listdir(INPUT_DIR):
    category_path = os.path.join(INPUT_DIR, category)
    if not os.path.isdir(category_path):
        continue

    for img_name in tqdm(os.listdir(category_path), desc=f"{category}"):
        img_path = os.path.join(category_path, img_name)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                ratio = width / height if height > 0 else 0
                data.append({
                    "class": category,
                    "filename": img_name,
                    "width": width,
                    "height": height,
                    "ratio": ratio
                })
        except Exception as e:
            print(f"Error con {img_name}: {e}")

# Crear DataFrame
df = pd.DataFrame(data)

# Guardar datos en CSV
df.to_csv("dataset_info.csv", index=False)
print(f"\nDatos guardados en dataset_info.csv ({len(df)} imágenes)\n")

# ------------------ #
#   Estadísticas     #
# ------------------ #
print("Resumen general:")
print(df.describe())

# Conteo por clase
class_counts = df["class"].value_counts()
print("\nCantidad de imágenes por clase:")
print(class_counts)

# Estadísticas por clase
stats = df.groupby("class")[["width", "height"]].agg(["mean", "min", "max"]).round(1)
print("\nTamaños promedio por clase (px):")
print(stats)

# ------------------ #
#      Gráficos      #
# ------------------ #

#  Cantidad de imágenes por clase
plt.figure(figsize=(8,5))
class_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Cantidad de imágenes por clase")
plt.xlabel("Clase")
plt.ylabel("Número de imágenes")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=200)
plt.close()

# Distribución de anchos y altos
plt.figure(figsize=(8,5))
plt.hist(df["width"], bins=30, alpha=0.7, label="Ancho")
plt.hist(df["height"], bins=30, alpha=0.7, label="Alto")
plt.title("Distribución de tamaños de imagen")
plt.xlabel("Pixeles")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.savefig("size_distribution.png", dpi=200)
plt.close()

# Proporciones ancho/alto
plt.figure(figsize=(8,5))
plt.hist(df["ratio"], bins=30, color="lightgreen", edgecolor="black")
plt.title("Relación ancho/alto (proporciones)")
plt.xlabel("Ratio (width / height)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("aspect_ratio_distribution.png", dpi=200)
plt.close()

print("Gráficos guardados:")
print(" - class_distribution.png")
print(" - size_distribution.png")
print(" - aspect_ratio_distribution.png")
