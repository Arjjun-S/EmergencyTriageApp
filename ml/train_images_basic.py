"""
Basic image training script for skin/rash classification.
Produces labels and a Keras model, then you can convert to .tflite.
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 15

DATA_DIR = os.path.join('..', 'datasets', 'SkinDisease')
ASSETS_DIR = os.path.join('..', 'app', 'src', 'main', 'assets', 'models')
os.makedirs(ASSETS_DIR, exist_ok=True)

train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='training'
)

val = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation'
)

model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train.num_classes, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=EPOCHS)

# Save Keras model and labels
keras_path = os.path.join(ASSETS_DIR, 'skin_cnn.h5')
model.save(keras_path)
labels_path = os.path.join(ASSETS_DIR, 'rash_model_labels.txt')
with open(labels_path, 'w', encoding='utf-8') as f:
    for name, idx in sorted(train.class_indices.items(), key=lambda x: x[1]):
        f.write(f"{name}\n")

print(f"Saved model -> {keras_path}")
print(f"Saved labels -> {labels_path}")
