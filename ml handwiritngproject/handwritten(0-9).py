import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=3)

# Save the model in .keras format
model.save('handwritten_model.keras', save_format='keras_v3')

model = tf.keras.models.load_model('handwritten_model.keras')

# Uncomment if you need to evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

image_number = 1
while os.path.isfile(f"images/digit{image_number}.png"):
    try:
        # Read the image in grayscale
        img = cv2.imread(f"images/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read image {image_number}")
            image_number += 1
            continue

        # Resize the image to 28x28 pixels (if necessary)
        img = cv2.resize(img, (28, 28))

        # Invert the image colors
        img = np.invert(img)

        # Reshape the image to match the input shape required by the model
        img = img.reshape(1, 28, 28, 1)

        # Normalize the image
        img = img / 255.0

        # Make a prediction
        prediction = model.predict(img)

        # Print the result
        print(f"This digit is probably a {np.argmax(prediction)}")

        # Display the image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        image_number += 1