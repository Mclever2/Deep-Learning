import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def crear_modelo_cnn2d_lstm(input_shape, num_clases):
    frames, alto, ancho, canales = input_shape

    input_layer = layers.Input(shape=input_shape)

    base_cnn = MobileNetV2(include_top=False, weights='imagenet', input_shape=(alto, ancho, canales))
    base_cnn.trainable = False  

    base_cnn = models.Model(inputs=base_cnn.input, outputs=layers.GlobalAveragePooling2D()(base_cnn.output))

    x = layers.TimeDistributed(base_cnn)(input_layer)

    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(num_clases, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def entrenar_modelo(model, X_train, y_train, X_val, y_val, epochs=4, batch_size=8):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return history
