"""Model Factory for Transfer Learning classifiers.

Supports VGG16, ResNet50, and EfficientNetB0 with ImageNet pre-trained weights.
"""

import logging
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    EfficientNetB0,
    ResNet50,
    VGG16,
)

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "vgg16": VGG16,
    "resnet50": ResNet50,
    "efficientnetb0": EfficientNetB0,
}


def create_model(
    model_name: str,
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    freeze_base: bool = True,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
) -> keras.Model:
    """Create a transfer learning model.

    Args:
        model_name: Name of the base model (vgg16, resnet50, efficientnetb0).
        num_classes: Number of output classes.
        input_shape: Input image dimensions.
        freeze_base: Whether to freeze base model weights.
        dropout_rate: Dropout rate for regularization.
        learning_rate: Learning rate for optimizer.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If model_name is not supported.
    """
    model_name_lower = model_name.lower()
    if model_name_lower not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    logger.info(f"Creating {model_name} model with {num_classes} classes")

    base_model = MODEL_REGISTRY[model_name_lower](
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    if freeze_base:
        base_model.trainable = False
        logger.info(f"Froze {len(base_model.layers)} base layers")

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "categorical_crossentropy"

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"tl_{model_name_lower}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    total_params = model.count_params()
    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    return model


def unfreeze_model(
    model: keras.Model,
    num_layers: Optional[int] = None,
    learning_rate: float = 1e-5,
) -> keras.Model:
    """Unfreeze base model layers for fine-tuning.

    Args:
        model: Compiled Keras model.
        num_layers: Number of layers to unfreeze from the top. None = all.
        learning_rate: Reduced learning rate for fine-tuning.

    Returns:
        Re-compiled model with unfrozen layers.
    """
    base_model = model.layers[1]
    base_model.trainable = True

    if num_layers is not None:
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
        logger.info(f"Unfroze top {num_layers} layers")
    else:
        logger.info("Unfroze all base model layers")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=model.loss,
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    return model
