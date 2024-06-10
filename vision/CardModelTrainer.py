import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model.Shape import ShapeType
from model.Color import ColorType
from model.Number import Number
from model.Shading import ShadingType
from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import os

# Define constants
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50

#This helps us artifically expand the training dataset by creating variations of the existing images
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

class CardModelTrainer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model_dir = 'models'        
        os.makedirs(self.model_dir, exist_ok=True)


    def preprocess_image(self, image):
        resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        normalized = resized / 255.0
        return normalized.flatten()
    
    def augment_image(self, image):
        # Apply data augmentation to the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        images_aug = next(datagen.flow(image, batch_size=1))  # Apply augmentations
        return images_aug[0]  # Remove batch dimension

    
    def load_dataset(self):
        data = pd.read_csv(self.csv_path)
        shape_images = []
        color_histograms = []
        number_images = []
        shading_images = []
        shape_labels = []
        color_labels = []
        number_labels = []
        shading_labels = []

        for _, row in data.iterrows():
            image_path = row['image_path']
            image = cv2.imread(image_path)
            if image is not None:
                # Shape classification
                shape_images.append(self.preprocess_image(image))
                shape_labels.append(row['shape'])
                # Color classification
                color_histograms.append(self.preprocess_color_image(image))
                color_labels.append(row['color'])
                # Number and shading classification
                number_images.append(self.preprocess_image(image))
                number_labels.append(row['number'])
                shading_images.append(self.preprocess_image(image))
                shading_labels.append(row['shading'])


        shape_images = np.array(shape_images).reshape((-1, 128, 128, 3))
        color_histograms = np.array(color_histograms).reshape((-1, 256))
        number_images = np.array(number_images).reshape((-1, 128, 128, 3))
        shading_images = np.array(shading_images).reshape((-1, 128, 128, 3))

        
        return (
            np.array(shape_images), shape_labels,
            np.array(color_histograms), color_labels,
            np.array(number_images), number_labels,
            np.array(shading_images), shading_labels
        )


    def map_labels_to_enum(self, labels, enum_class):
        return [enum_class[label.strip()].value for label in labels]
    
    def create_model_with_transfer_learning(self, input_shape, num_classes):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    def preprocess_color_image(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    
    def train_color_model(self, color_histograms, color_labels):
        color_labels = self.map_labels_to_enum(color_labels, ColorType)  # Do not use to_categorical
        X_train, X_test, y_train, y_test = train_test_split(color_histograms, color_labels, test_size=0.2, random_state=42)

        # Ensure the data type is float32
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int32)

        num_classes = len(ColorType)      # Create and compile the model
        model = Sequential([
            Input(shape=(256,)),  # Use the Input layer explicitly
            Dense(128, activation='relu'),  
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')  # Ensure the output layer has the correct number of classes
        ])
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

        # Save the model
        model.save('models/color_model_fine_tuned.keras')

        return model


    # Train the shape model
    def train_shape_model(self, images, shape_labels):
        shape_labels = to_categorical(self.map_labels_to_enum(shape_labels, ShapeType))
        X_train, X_test, y_train, y_test = train_test_split(images, shape_labels, test_size=0.2, random_state=42)

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(X_train)

        # Ensure the output layer matches the number of unique classes in shape_labels
        num_classes = shape_labels.shape[1]
        model = self.create_model_with_transfer_learning(input_shape=(128, 128, 3), num_classes=num_classes)
        model_checkpoint = ModelCheckpoint('models/shape_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        model.fit(datagen.flow(X_train, y_train, batch_size=32),
                  epochs=50, validation_data=(X_test, y_test),
                  callbacks=[early_stopping, model_checkpoint, reduce_lr])

        for layer in model.layers[-4:]:
            layer.trainable = True

        model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(datagen.flow(X_train, y_train, batch_size=32),
                  epochs=20, validation_data=(X_test, y_test),
                  callbacks=[early_stopping, model_checkpoint, reduce_lr])

        model.save('models/shape_model_fine_tuned.keras')

    # Train the shading model
    def train_shading_model(self, images, shading_labels):
        shading_labels = to_categorical(self.map_labels_to_enum(shading_labels, ShadingType))
        X_train, X_test, y_train, y_test = train_test_split(images, shading_labels, test_size=0.2, random_state=42)

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(X_train)

        # Ensure the output layer matches the number of unique classes in shading_labels
        num_classes = shading_labels.shape[1]
        model = self.create_model_with_transfer_learning(input_shape=(128, 128, 3), num_classes=num_classes)
        model_checkpoint = ModelCheckpoint('models/shading_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        model.fit(datagen.flow(X_train, y_train, batch_size=32),
                  epochs=50, validation_data=(X_test, y_test),
                  callbacks=[early_stopping, model_checkpoint, reduce_lr])

        for layer in model.layers[-4:]:
            layer.trainable = True

        model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(datagen.flow(X_train, y_train, batch_size=32),
                  epochs=20, validation_data=(X_test, y_test),
                  callbacks=[early_stopping, model_checkpoint, reduce_lr])

        model.save('models/shading_model_fine_tuned.keras')

    def train_models(self):
        shape_images, shape_labels, color_histograms, color_labels, number_images, number_labels, shading_images, shading_labels = self.load_dataset()
        
        # Train individual models for each attribute
        self.train_shape_model(shape_images, shape_labels)
        self.train_shading_model(shading_images, shading_labels)
        self.train_color_model(color_histograms, color_labels)