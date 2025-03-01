import os

# Clone the PlantVillage dataset from GitHub (if not already cloned)
repo_url = 'https://github.com/spMohanty/PlantVillage-Dataset.git'
clone_dir = 'PlantVillage-Dataset'

# Check if the repository already exists; if not, clone it
if not os.path.exists(clone_dir):
    os.system(f'git clone {repo_url}')
else:
    print(f'Repository {clone_dir} already exists.')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to the dataset (adjust the paths based on the cloned directory)
data_dir = os.path.join('PlantVillage-Dataset', 'path_to_your_dataset_folder')  # Adjust this path

# Set up ImageDataGenerator with augmentation for training and rescaling for testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data from the 'train' directory
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),  # Adjust this to match the structure
    target_size=(150, 150),  # Resize images
    batch_size=32,
    class_mode='categorical'
)

# Load validation data from the 'test' directory
validation_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),  # Adjust this to match the structure
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Load the pre-trained VGG16 model (without the top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the base layers to avoid re-training them
base_model.trainable = False

# Add custom layers for classification
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)