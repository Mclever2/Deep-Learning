import os
import cv2

def analizar_dataset(dataset_path, esperado_frames=60, resolucion_esperada=(96, 96)):
    resumen = {}
    errores = []

    for clase in sorted(os.listdir(dataset_path)):
        clase_path = os.path.join(dataset_path, clase)
        if not os.path.isdir(clase_path):
            continue

        resumen[clase] = {
            "total": 0,
            "frames_correctos": 0,
            "frames_incorrectos": 0,
            "resolucion_correcta": 0,
            "resolucion_incorrecta": 0
        }

        for archivo in os.listdir(clase_path):
            if not archivo.endswith(".avi"):
                continue

            path_video = os.path.join(clase_path, archivo)
            cap = cv2.VideoCapture(path_video)
            if not cap.isOpened():
                errores.append((clase, archivo, "no se puede abrir"))
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if frame_count == esperado_frames:
                resumen[clase]["frames_correctos"] += 1
            else:
                resumen[clase]["frames_incorrectos"] += 1
                errores.append((clase, archivo, f"{frame_count} frames"))

            if (width, height) == resolucion_esperada:
                resumen[clase]["resolucion_correcta"] += 1
            else:
                resumen[clase]["resolucion_incorrecta"] += 1
                errores.append((clase, archivo, f"resolución {width}x{height}"))

            resumen[clase]["total"] += 1
            cap.release()

            print(f"[{clase}] {archivo}: {frame_count} frames, {width}x{height}px, {fps:.1f} fps")

    print("\nResumen por clase:")
    for clase, datos in resumen.items():
        print(f"{clase}: {datos['total']} videos "
              f"({datos['frames_correctos']} frames OK, {datos['resolucion_correcta']} resolución OK)")

    if errores:
        print("\nVideos con errores:")
        for clase, archivo, motivo in errores:
            print(f"{clase}/{archivo}: {motivo}")
    else:
        print("\nNo se detectaron errores.")

if __name__ == "__main__":
    analizar_dataset("dataset", esperado_frames=60, resolucion_esperada=(96, 96))
