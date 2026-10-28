from pathlib import Path
import cv2
from ultralytics import YOLO

# 1. Definir carpetas relativas directas
CARPETA_IMAGENES = Path("imagenes_convertir")
CARPETA_LABELS = Path("nuevos_labels_pose")
CARPETA_LABELS.mkdir(parents=True, exist_ok=True)

# 2. Cargar el modelo base de Pose
model = YOLO("yolo11n-pose.pt")

# Extensiones de imagen soportadas
extensiones = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.PNG")
imagenes = []
for ext in extensiones:
    imagenes.extend(CARPETA_IMAGENES.glob(ext))

print(f"Total de imágenes encontradas: {len(imagenes)}")

procesadas = 0

for ruta_img in imagenes:
    img = cv2.imread(str(ruta_img))
    if img is None:
        continue

    alto, ancho, _ = img.shape

    # Inferencia con umbral sensible
    results = model(img, conf=0.35, verbose=False)

    lineas_guardar = []

    for r in results:
        if r.boxes is not None and r.keypoints is not None:
            cajas = r.boxes.xywhn.cpu().numpy()  # Cajas normalizadas (x_c, y_c, w, h)
            puntos = r.keypoints.xyn.cpu().numpy()  # Puntos normalizados (x, y)
            confs_kpts = (
                r.keypoints.conf.cpu().numpy()
                if r.keypoints.conf is not None
                else None
            )

            for idx, box in enumerate(cajas):
                # CLASE 1: Se fuerza directamente a 'encapuchado'
                clase = 1
                bx, by, bw, bh = box

                linea = [f"{clase}", f"{bx:.6f}", f"{by:.6f}", f"{bw:.6f}", f"{bh:.6f}"]

                # Extraer los 17 puntos COCO (X, Y, Visibilidad)
                pts = puntos[idx]
                for p_idx, (px, py) in enumerate(pts):
                    vis = 2  # Visible
                    if confs_kpts is not None:
                        conf_val = confs_kpts[idx][p_idx]
                        if conf_val < 0.25:
                            vis = 0  # No visible / oculto
                        elif conf_val < 0.5:
                            vis = 1  # Parcialmente visible

                    if px == 0 and py == 0:
                        vis = 0

                    linea.extend([f"{px:.6f}", f"{py:.6f}", f"{vis}"])

                lineas_guardar.append(" ".join(linea) + "\n")

    # Guardar archivo .txt con el mismo nombre de la imagen
    archivo_txt = CARPETA_LABELS / f"{ruta_img.stem}.txt"
    with open(archivo_txt, "w", encoding="utf-8") as f:
        f.writelines(lineas_guardar)

    procesadas += 1

print(f"¡Completado! Se generaron {procesadas} archivos de etiquetas con clase '1' en '{CARPETA_LABELS}'.")