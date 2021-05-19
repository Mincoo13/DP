import keras
from keras import models, layers

def set_model(input_shape, X_train, y_train, X_val, y_val):
    """
    Funkcia, v ktorej sa definuje model a jeho vrstvy spolu s loss funkciou a
    optimizerom.
    Vrstva, ktora bude pouzita na nasledne vyhodnotenie map pozornosti musi mat
    nazov 'heatmap_layer'
    """
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.4))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), name='heatmap_layer'))
    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Dropout(rate=0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(3, activation='softplus'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy'])
    
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                                restore_best_weights=True)
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #                                               patience=4, min_lr=0.001)

    history = model.fit(X_train, y_train, callbacks=[es_callback], epochs=40,
                        validation_data=[X_val, y_val])

    return model, history
