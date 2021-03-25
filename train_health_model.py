import pickle
from keras import models, layers

def set_model(input_shape, X_train, y_train, X_val, y_val):
    """
    Funkcia, v ktorej sa definuje model a jeho vrstvy spolu s loss funkciou a
    optimizerom.
    Vrstva, ktora bude pouzita na nasledne vyhodnotenie map pozornosti musi mat
    nazov 'heatmap_layer'
    """
    model = models.Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                            input_shape=input_shape))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), name='heatmap_layer'))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(3, activation='softmax'))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=2,
                        validation_data=[X_val, y_val])

    return model, history
