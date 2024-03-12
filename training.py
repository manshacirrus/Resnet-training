import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

# Define paths to the dataset
base_dir = r"C:\Users\Asus\Downloads\Microsoft COCO.v2-raw.coco"  # Update this path
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

# Parameters
img_height, img_width = 224, 224  # ResNet50 default image size
batch_size = 32

# Preprocessing function for ResNet50
def preprocess_data(img):
    return preprocess_input(img)

# Load and preprocess the dataset
train_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_data)
valid_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_data)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Load ResNet50 model, excluding the top layer
input_tensor = Input(shape=(img_height, img_width, 3))
base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # New FC layer, random init
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # New softmax layer

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=epochs
)

# Save the model
model.save('resnet50_coco_object_detection_model.h5')
