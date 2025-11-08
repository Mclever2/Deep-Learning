import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque

frames_totales = 30
video_size = 64
modelo_path = "modelo_cnn2d_lstm_30_64.keras"
encoder_path = "label_encoder.pkl"
frecuencia_prediccion = 30  

model = tf.keras.models.load_model(modelo_path)
with open(encoder_path, "rb") as f:
    le = pickle.load(f)

cap = cv2.VideoCapture(0)
buffer_frames = deque(maxlen=frames_totales)
frame_count = 0
ultima_prediccion = "Esperando..."

print("Presiona 'ESC' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    region = frame_rgb[y0:y1, x0:x1]
    frame_resized = cv2.resize(region, (video_size, video_size))
    frame_norm = frame_resized.astype(np.float32) / 255.0
    buffer_frames.append(frame_norm)

    texto = ""

    if len(buffer_frames) < frames_totales:
        texto = f"Esperando... ({len(buffer_frames)}/{frames_totales})"
    else:
        frame_count += 1

        if frame_count % frecuencia_prediccion == 0:
            texto = "Traduciendo..."
            input_frames = np.array(buffer_frames)
            input_frames = np.expand_dims(input_frames, axis=0) 

            prediction = model.predict(input_frames, verbose=0)[0]
            pred_clase = le.inverse_transform([np.argmax(prediction)])[0]
            conf = np.max(prediction)
            ultima_prediccion = f"Prediccion: {pred_clase} ({conf*100:.1f}%)"
        else:
            texto = "Traduciendo..."

    cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2)

    cv2.putText(frame, ultima_prediccion, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    cv2.imshow("Clasificacion en tiempo real", frame)
    if cv2.waitKey(1) == 27:  
        break

cap.release()
cv2.destroyAllWindows()
2