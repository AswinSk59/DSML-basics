import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10 from PIL import Image
import numpy as np
(X_train, y_train), (X_test, y_test) = cifar10.load_data() X_train, X_test = X_train / 255.0, X_test / 255.0




X_train, X_test = X_train / 255.0, X_test / 255.0


model = keras.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'), layers.Flatten(),
layers.Dense(64, activation='relu'), layers.Dense(10) # 10 output classes
])


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test))
 
 





test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2) print("\nTest accuracy:", test_ac)
class_names = [
"Airplane", "Automobile", "Bird",
"Cat",
"Deer",
"Dog",
"Frog",
"Horse",
"Ship", "Truck"
]
# Load and preprocess the image image_path = '0007.jpeg'
image = Image.open(image_path).resize((32, 32)) image = np.array(image) / 255.0
image = np.expand_dims(image, axis=0)


# Make predictions
predictions = model.predict(image)




# Get the predicted class index
 
predicted_class_index = np.argmax(predictions)


# Get the class name from the class names list predicted_class_name = class_names[predicted_class_index] print(f'Predicted class: {predicted_class_name}')
