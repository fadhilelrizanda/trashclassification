import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
import keras
from PIL import Image
import wandb
from wandb.integration.keras import WandbMetricsLogger,WandbModelCheckpoint

def main():
    wandb.login(KEY_WANDB)
    total_data_class = []
    base_path = "/dataset-original"
    all_dir_class = os.listdir(base_path)


    all_dir_class.remove(".DS_Store")
    total_data_class = []
    for img_data in all_dir_class:
        total_data_class.append(len(os.listdir(os.path.join(base_path,img_data))))


    batch_size = 32
    img_height = 180
    img_width = 180
    data_dir = ("/content/dataset-original")

    train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=(img_height, img_width),
            batch_size=batch_size)

    # Define data augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),

    ])

    # Apply data augmentation to the training dataset
    augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))


    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    # Configure the datasets for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    augmented_train_ds = augmented_train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


    wandb.init(
        # set the wandb project where this run will be logged
        project="project1",

        # track hyperparameters and run metadata with wandb.config
        config={
            "layer_1": 512,
            "activation_1": "relu",
            "dropout_1": 0.2,
            "layer_2": 6,
            "activation_2": "softmax",
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metric": "accuracy",
            "epoch": 5,
            "batch_size": 64
        }
    )
    config = wandb.config
    # Building Model

    batch_size = 64
    num_epochs = 10
    image_height, image_width = 180, 180  # Adjust according to your image size

    # Load EfficientNetB0 model pre-trained on ImageNet, excluding top layers
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))

    # Freeze the base model
    base_model.trainable = False


    # Define the CNN model
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(config.layer_1,activation= config.activation_1),
        keras.layers.Dropout(config.dropout_1),
        keras.layers.Dense(config.layer_2, activation=config.activation_2)
    ])

    model2 = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(config.layer_1,activation= config.activation_1),
        keras.layers.Dropout(config.dropout_1),
        keras.layers.Dense(config.layer_2, activation=config.activation_2)
    ])

    # Compile the model
    model.compile(optimizer=config.optimizer,
                loss=config.loss,
                metrics=[config.metric])


    history = model.fit(augmented_train_ds,
                        epochs=config.epoch,validation_data = val_ds,
                        callbacks=wandb_callbacks,
                        batch_size=config.batch_size,
                    )

    # Compile the model
    model2.compile(optimizer=config.optimizer,
                loss=config.loss,
                metrics=[config.metric])

    class_names = train_ds.class_names
    label_counts = {class_name: 0 for class_name in class_names}
    for images, labels in train_ds.unbatch():
        label_counts[class_names[labels.numpy()]]+=1



    total_samples = sum(label_counts.values())
    class_weights = {i: total_samples / (len(class_names) * count) for i, count in enumerate(label_counts.values())}

    history2 = model2.fit(augmented_train_ds,
                        epochs=config.epoch,validation_data = val_ds,
                        callbacks=wandb_callbacks,
                        batch_size=config.batch_size,
                        class_weight=class_weights)

    wandb.finish()

    model.save("Model_1")
    model2.save("Model_2")


if __name__: "__main__":
    main()
