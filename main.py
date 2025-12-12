import tensorflow as tf
import numpy as np
from dataset_loader import cargar_dataset
from preparacion_datos import dividir_por_persona, preparar_etiquetas
from modelo import crear_modelo_cnn2d_lstm, entrenar_modelo  
from visualizations import plot_training_history, plot_confusion_matrix  

if __name__ == "__main__":

    # Configuración para evitar errores de memoria fragmentada
    # (Opcional, pero ayuda en Windows con GPUs de poca memoria)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    dataset_path = "dataset"
    frames_totales = 60
    video_size = 96

    print("--- FASE 1: CARGA EN CPU ---")
    print("Cargando dataset (Optimizacion float16 activada)...")
    # Los datos se quedan en RAM del sistema (CPU)
    X, y, rutas, le = cargar_dataset(dataset_path, frames_totales, video_size)

    print(f"Dataset en RAM. Shape: {X.shape}, Tipo: {X.dtype}")

    print("Dividiendo dataset...")
    X_train, X_val, y_train_int, y_val_int = dividir_por_persona(X, y, rutas)

    print("Procesando etiquetas...")
    num_classes = len(le.classes_)
    y_train = preparar_etiquetas(y_train_int, num_clases=num_classes)
    y_val = preparar_etiquetas(y_val_int, num_clases=num_classes)
    
    # Aseguramos float16 en etiquetas también
    y_train = y_train.astype(np.float16)
    y_val = y_val.astype(np.float16)

    print("--- FASE 2: CONFIGURACIÓN DE GENERADORES (STREAMING) ---")
    
    # DEFINICIÓN DE GENERADORES
    # Esto evita copiar los 2GB a la GPU de golpe. 
    # La GPU pedirá los datos poco a poco desde la RAM.
    
    def generator_train():
        for i in range(len(X_train)):
            yield X_train[i], y_train[i]

    def generator_val():
        for i in range(len(X_val)):
            yield X_val[i], y_val[i]

    # Definimos la estructura de los datos para TensorFlow
    output_signature = (
        tf.TensorSpec(shape=(frames_totales, video_size, video_size, 3), dtype=tf.float16),
        tf.TensorSpec(shape=(num_classes,), dtype=tf.float16)
    )

    BATCH_SIZE = 4 # Mantenemos batch pequeño para la GTX 1650

    # Creación del dataset usando el generador
    train_dataset = tf.data.Dataset.from_generator(
        generator_train,
        output_signature=output_signature
    )
    # Importante: repeat() asegura que el generador no se agote tras la primera época
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        generator_val,
        output_signature=output_signature
    )
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("Generadores listos. La GPU no se sobrecargará.")

    print("--- FASE 3: MODELO Y ENTRENAMIENTO ---")
    print("Creando modelo CNN2D + LSTM con MobileNetV2...")
    input_shape = (frames_totales, video_size, video_size, 3)
    model = crear_modelo_cnn2d_lstm(input_shape, num_clases=num_classes)  

    print("Iniciando entrenamiento...")
    # El modelo pedirá datos al generador, batch por batch
    history = entrenar_modelo(model, train_dataset, val_dataset, epochs=6)

    print("Guardando modelo...")
    model.save("modelo_cnn2d_lstm_30_96.keras")  
    print("Modelo guardado exitosamente.")

    plot_training_history(history)
    # Nota: Pasamos los arrays numpy a la matriz de confusión (esto está bien, es en CPU)
    plot_confusion_matrix(model, X_val, y_val, le)