import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy

import numpy as np
import matplotlib.pyplot as plt
import os

class HairDiseaseDetector:
    def __init__(self, img_size=(224, 224), num_classes=11):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        # Disease classes mapping
        self.disease_classes = [
            'Alopecia Areata', 'Contact Dermatitis', 'Folliculitis', 
            'Head Lice', 'Healthy hair', 'Lichen Planus', 
            'Male Pattern Baldness', 'Psoriasis', 'Seborrheic Dermatitis', 
            'Telogen Effluvium', 'Tinea Capitis'
        ]
    
    def build_model(self):
        """Build EfficientNet model with custom head"""
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        return self.model
    
    def create_data_generators(self, data_dir, batch_size=32):
        """Create data generators for training, validation, and testing"""
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation and test data (only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            os.path.join(data_dir, 'val'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(data_dir, 'test'),
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def train_model(self, train_gen, val_gen, epochs=50, fine_tune_epochs=20):
        """Train the model with two-stage training"""
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'best_hair_disease_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        print("Phase 1: Training with frozen base model...")
        # Phase 1: Train with frozen base model
        history1 = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nPhase 2: Fine-tuning with unfrozen base model...")
        # Phase 2: Unfreeze and fine-tune
        self.model.get_layer('efficientnetb0').trainable = True
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        # Fine-tune training
        history2 = self.model.fit(
            train_gen,
            epochs=fine_tune_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
        
        return self.history
    
    def evaluate_model(self, test_gen):
        """Evaluate model on test set"""
        test_loss, test_acc, test_top3_acc = self.model.evaluate(test_gen, verbose=1)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Top-3 Accuracy: {test_top3_acc:.4f}")
        return test_loss, test_acc, test_top3_acc
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history['loss'], label='Training Loss')
        ax2.plot(self.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_disease(self, img_path):
        """Predict hair disease from image"""
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[::-1][:3]
        top_3_predictions = [(self.disease_classes[idx], predictions[0][idx]) for idx in top_3_idx]
        
        return {
            'predicted_disease': self.disease_classes[predicted_class_idx],
            'confidence': float(confidence),
            'top_3_predictions': top_3_predictions
        }
    
    def save_model(self, filepath='hair_disease_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='hair_disease_model.h5'):
        """Load a pre-trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """Main training function"""
    # Initialize detector
    detector = HairDiseaseDetector()
    
    # Build model
    model = detector.build_model()
    print("Model architecture:")
    model.summary()
    
    # Set data directory path
    data_dir = r"C:\CODING\AI\PROJECTS\HC\Models\FinalDataset\fd\Rotation\Gaussian"  # Adjust path as needed
    
    # Create data generators
    train_gen, val_gen, test_gen = detector.create_data_generators(data_dir, batch_size=32)
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Number of classes: {train_gen.num_classes}")
    
    # Train model
    history = detector.train_model(train_gen, val_gen, epochs=30, fine_tune_epochs=15)
    
    # Evaluate model
    detector.evaluate_model(test_gen)
    
    # Plot training history
    detector.plot_training_history()
    
    # Save model
    detector.save_model('final_hair_disease_model.h5')
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()