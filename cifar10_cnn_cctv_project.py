import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from tensorflow.keras.datasets import cifar10

# LOAD CIFAR-10 DATASET

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

# SCALE IMAGES


X_train_scale = X_train / 255.0
X_test_scale = X_test / 255.0

# CIFAR-10 CLASS NAMES

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# BUILD CNN MODEL

cnn_model = Sequential()

cnn_model.add(Input(shape=(32, 32, 3)))

cnn_model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    activation="relu"
))

cnn_model.add(MaxPooling2D(
    pool_size=(2, 2),
    padding="same"
))

cnn_model.add(Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    activation="relu"
))

cnn_model.add(MaxPooling2D(
    pool_size=(2, 2),
    padding="same"
))

cnn_model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    activation="relu"
))

cnn_model.add(MaxPooling2D(
    pool_size=(2, 2),
    padding="same"
))

cnn_model.add(Flatten())

cnn_model.add(Dense(128, activation="relu"))
cnn_model.add(Dense(256, activation="relu"))
cnn_model.add(Dense(512, activation="relu"))
cnn_model.add(Dense(1024, activation="relu"))

cnn_model.add(Dense(10, activation="softmax"))

cnn_model.summary()

# COMPILE MODEL

cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# TRAIN MODEL

print("\nTraining Started...\n")

history = cnn_model.fit(
    X_train_scale,
    y_train,
    epochs=10,
    validation_split=0.1
)

# EVALUATE MODEL

print("\nEvaluating Model...\n")

loss, accuracy = cnn_model.evaluate(
    X_test_scale,
    y_test,
    verbose=1
)

print("\n==============================")
print("Test Loss     :", loss)
print("Test Accuracy :", accuracy)
print("==============================")

# PLOT ACCURACY GRAPH

plt.figure(figsize=(8, 5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("CNN Model Accuracy on CIFAR-10")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# =====================================================
# WEBCAM / CCTV OBJECT CLASSIFICATION
# =====================================================

print("\nStarting Webcam Detection...")
print("Press 'q' to Quit\n")

cap = cv2.VideoCapture(0)

# For CCTV:
# cap = cv2.VideoCapture("YOUR_CCTV_STREAM_URL")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to access camera")
        break

    # Resize frame to CIFAR-10 size
    img = cv2.resize(frame, (32, 32))

    # Normalize
    img = img / 255.0

    # Reshape for CNN
    img = img.reshape(1, 32, 32, 3)

    # Predict
    prediction = cnn_model.predict(
        img,
        verbose=0
    )

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    label = class_names[class_index]

    # Display result on webcam
    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Print prediction in terminal
    print(
        f"Prediction: {label} | Confidence: {confidence:.4f}"
    )

    cv2.imshow(
        "CIFAR10 Object Classification",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nProgram Finished Successfully")
