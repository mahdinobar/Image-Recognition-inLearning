import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from pathlib import Path

# Load data set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data set to 0-to-1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create a model and add layers
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation= 'relu', input_shape=(32, 32, 3)))
#input is not an image so we have no padding
model.add(Conv2D(32, (3, 3), activation= 'relu'))
#to speed up the process we add max pool
model.add(MaxPooling2D(pool_size=(2,2)))
#to prevent neuron co-adaptation to avoid overfitting
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#to tell Dense layer that we are no longer on 2D data (do this flatten layer whenever you go from convolutional to dense layer)
model.add(Flatten())

model.add(Dense(512, activation="relu"))
#to let NN work hard to make correct output by 50 percent dropout
model.add(Dropout(0.50))
model.add(Dense(10, activation="softmax"))

# Compile NN by Keras
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=30,
    validation_data=(x_test, y_test),
    shuffle=True
)

# Save NN structure
model_structure = model.to_json()
f = Path('model_structure.json')
f.write_text(model_structure)

# Save NN weights
model.save_weights('model_weights.h5')

# Print a summary of the model
model.summary()