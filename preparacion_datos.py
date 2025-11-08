from tensorflow.keras.utils import to_categorical
import numpy as np

def dividir_por_persona(X, y, rutas, test_size=0.2, random_state=42):

    if len(X) == 0:
        raise ValueError("El dataset está vacío")

    personas = []
    for nombre in rutas:
        try:
            persona = nombre.split("_")[1]  
            personas.append(persona)
        except IndexError:
            personas.append("DESCONOCIDO")

    personas_unicas = list(set(personas))
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(personas_unicas)

    num_train = int(len(personas_unicas) * (1 - test_size))
    personas_train = set(personas_unicas[:num_train])
    personas_val = set(personas_unicas[num_train:])

    X_train, y_train, X_val, y_val = [], [], [], []
    for xi, yi, persona in zip(X, y, personas):
        if persona in personas_train:
            X_train.append(xi)
            y_train.append(yi)
        else:
            X_val.append(xi)
            y_val.append(yi)

    print(f"Total de muestras: {len(X)}")
    print(f"Entrenamiento: {len(X_train)} - Validación: {len(X_val)}")

    return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)


def preparar_etiquetas(y, num_clases):

    return to_categorical(y, num_classes=num_clases)
