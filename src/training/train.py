"""Training pipeline for Transfer Learning image classifier."""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.models.model_factory import create_model, unfreeze_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_generators(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """Create training and validation data generators with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, val_generator


def get_callbacks(
    model_dir: str,
    patience: int = 5,
) -> list:
    """Get training callbacks."""
    os.makedirs(model_dir, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, "logs"),
        ),
    ]


def train(
    model_name: str = "resnet50",
    data_dir: str = "data/",
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    fine_tune_epochs: int = 10,
    fine_tune_layers: int = 30,
    model_dir: str = "models/",
) -> keras.Model:
    """Full training pipeline with two-phase transfer learning."""
    logger.info(f"Starting training with {model_name}")

    train_gen, val_gen = create_data_generators(
        data_dir, batch_size=batch_size
    )
    num_classes = train_gen.num_classes
    logger.info(f"Found {num_classes} classes: {list(train_gen.class_indices.keys())}")

    # Phase 1: Train with frozen base
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )

    callbacks = get_callbacks(model_dir)

    logger.info("Phase 1: Training classification head...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Phase 2: Fine-tune
    if fine_tune_epochs > 0:
        logger.info(f"Phase 2: Fine-tuning top {fine_tune_layers} layers...")
        model = unfreeze_model(
            model,
            num_layers=fine_tune_layers,
            learning_rate=learning_rate / 10,
        )
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
        )

    # Save final model
    final_path = os.path.join(model_dir, f"{model_name}_final.keras")
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transfer Learning classifier")
    parser.add_argument("--model", type=str, default="resnet50", choices=["vgg16", "resnet50", "efficientnetb0"])
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-dir", type=str, default="models/")
    args = parser.parse_args()

    train(
        model_name=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_dir=args.model_dir,
    )
