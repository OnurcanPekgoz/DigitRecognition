from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


def create_cnn_model():
    model = Sequential()  # Input size = (28,28,1)
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Output size = (26,26,32)
    model.add(MaxPooling2D((2, 2)))  # Output size = (13,13,32)
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))  # Output size = (11,11,64)
    model.add(MaxPooling2D((2, 2)))  # Output size = (5,5,64)
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))  # Output size = (3,3,64)
    model.add(Flatten())  # Output size = 3*3*64=576
    model.add(Dense(128, activation='relu'))  # Output size =128
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))  # Output size = 64
    model.add(Dense(10, activation='softmax'))  # Output size = 10

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
y_train = to_categorical(y_train)

cnn_model = create_cnn_model()
cnn_model.fit(x_train, y_train, epochs=10, batch_size=64)

save_model(cnn_model, 'mnist_cnn_model.h5')

x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0
y_test = to_categorical(y_test)

trained_model = load_model('mnist_cnn_model.h5')

eval_result = trained_model.evaluate(x_test, y_test)

print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")
