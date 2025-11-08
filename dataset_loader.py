import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

def cargar_video(path, frames_totales=30, video_size=64):
    cap = cv2.VideoCapture(path)
    frames = []

    while len(frames) < frames_totales:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (video_size, video_size))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) != frames_totales:
        return None

    return np.array(frames)

def cargar_dataset(dataset_path, frames_totales=60, video_size=96):
    X = []
    etiquetas_texto = []
    rutas = []  

    for clase in sorted(os.listdir(dataset_path)):
        clase_path = os.path.join(dataset_path, clase)
        if not os.path.isdir(clase_path):
            continue

        for archivo in os.listdir(clase_path):
            if archivo.endswith(".mp4"):
                path = os.path.join(clase_path, archivo)
                video = cargar_video(path, frames_totales, video_size)
                if video is not None:
                    X.append(video)
                    etiquetas_texto.append(clase)
                    rutas.append(archivo)  # solo el nombre, no toda la ruta

    le = LabelEncoder()
    y = le.fit_transform(etiquetas_texto)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return np.array(X), y, rutas, le