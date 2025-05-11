import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Base directories
BASE_DIR = os.path.join(os.getcwd(), 'dataset')
TRAIN_DIR = os.path.join(BASE_DIR, 'Train_Set_Folder')
VAL_DIR = os.path.join(BASE_DIR, 'Validation_Set_Folder')
TEST_DIR = os.path.join(BASE_DIR, 'Test_Set_Folder')

# Selected 5 plant species
selected_species = [
    'aloevera',
    'orange',
    'sweet potatoes',
    'watermelon',
    'pineapple'
]

# Function to create generators
def get_generator(directory, selected_classes, target_size=(224, 224), batch_size=32, shuffle=True):
    datagen = ImageDataGenerator(rescale=1./255)
    return datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        classes=selected_classes,
        class_mode='categorical',
        shuffle=shuffle
    )

# Creating data generators for train, validation, test
train_gen = get_generator(TRAIN_DIR, selected_species)
val_gen = get_generator(VAL_DIR, selected_species)
test_gen = get_generator(TEST_DIR, selected_species, shuffle=False)

# Print class mapping
print("Class indices:", train_gen.class_indices)

import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')  # 5 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest accuracy: {test_acc:.2f}")

# Optional: Save the model
model.save("plant_classifier_model.h5")

