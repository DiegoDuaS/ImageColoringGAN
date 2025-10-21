import os
from PIL import Image
from tqdm import tqdm

# Carpetas base
INPUT_DIR = "ImagesRaw"
OUTPUT_COLOR = "ImagesProcessed/color"
OUTPUT_GRAY = "ImagesProcessed/gray"

# Tamaño deseado
TARGET_SIZE = (128, 128)

# Crear carpetas de salida si no existen
os.makedirs(OUTPUT_COLOR, exist_ok=True)
os.makedirs(OUTPUT_GRAY, exist_ok=True)

# Contador de imágenes global
counter = 1
errors = 0

# Recorre todas las subcarpetas
for category in os.listdir(INPUT_DIR):
    category_path = os.path.join(INPUT_DIR, category)
    if not os.path.isdir(category_path):
        continue

    print(f"Procesando categoría: {category}")

    for img_name in tqdm(os.listdir(category_path)):
        img_path = os.path.join(category_path, img_name)
        try:
            with Image.open(img_path) as img:
                # Convertir a RGB por seguridad
                img = img.convert("RGB")

                # Mantener proporciones (resize con padding)
                img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
                new_img = Image.new("RGB", TARGET_SIZE, (0, 0, 0))
                paste_x = (TARGET_SIZE[0] - img.width) // 2
                paste_y = (TARGET_SIZE[1] - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))

                # Nombre nuevo: img00001.jpg
                new_name = f"img{counter:05d}.jpg"

                # Guardar color
                color_path = os.path.join(OUTPUT_COLOR, new_name)
                new_img.save(color_path)

                # Guardar escala de grises
                gray_img = new_img.convert("L")
                gray_path = os.path.join(OUTPUT_GRAY, new_name)
                gray_img.save(gray_path)

                counter += 1
        except Exception as e:
            errors += 1
            print(f"Error con {img_name}: {e}")

# Resumen final
total = counter - 1
print("\n--- Procesamiento completado ---")
print(f"Total de imágenes procesadas: {total}")
print(f"Dimensiones finales: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
if errors > 0:
    print(f"Errores durante el proceso: {errors}")
else:
    print("Sin errores detectados.")
