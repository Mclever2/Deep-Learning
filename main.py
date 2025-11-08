from dataset_loader import cargar_dataset
from preparacion_datos import dividir_por_persona, preparar_etiquetas
from modelo import crear_modelo_cnn2d_lstm, entrenar_modelo  
from visualizations import plot_training_history, plot_confusion_matrix  


if __name__ == "__main__":

    dataset_path = "dataset"
    frames_totales = 30
    video_size = 64 

    print("Cargando dataset...")
    X, y, rutas, le = cargar_dataset(dataset_path, frames_totales, video_size)

    print("Dividiendo dataset por persona (entrenamiento: R y V, validación: M)...")
    X_train, X_val, y_train_int, y_val_int = dividir_por_persona(X, y, rutas)

    print("Codificando etiquetas...")
    y_train = preparar_etiquetas(y_train_int, num_clases=len(le.classes_))
    y_val = preparar_etiquetas(y_val_int, num_clases=len(le.classes_))

    print("Creando modelo CNN2D + LSTM con MobileNetV2...")
    input_shape = (frames_totales, video_size, video_size, 3)
    model = crear_modelo_cnn2d_lstm(input_shape, num_clases=len(le.classes_))  

    print("Entrenando modelo...")
    history = entrenar_modelo(model, X_train, y_train, X_val, y_val)

    print("Guardando modelo...")
    model.save("modelo_cnn2d_lstm_30_64.keras")  
    print("Modelo guardado como modelo_cnn2d_lstm_30_64.keras")

    plot_training_history(history)
    plot_confusion_matrix(model, X_val, y_val, le)
