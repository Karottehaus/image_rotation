import os
import tensorflow as tf
import json
import h5py
import logging
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from settings import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE, SEED, VALIDATION_SPLIT

AUTOTUNE = tf.data.AUTOTUNE


# input_shape = (7, 7, 2048)
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def save_features_batch(dataset, base_model, feature_filename, label_filename, batch_size=64):
    with h5py.File(feature_filename, 'w') as feature_file, h5py.File(label_filename, 'w') as label_file:
        # Process first batch to get shapes
        for images, labels in dataset.take(1):
            features = base_model.predict(images)
            feature_shape = features.shape[1:]
            label_shape = labels.shape[1:]

            # Create resizable datasets
            feature_dataset = feature_file.create_dataset(
                'features',
                shape=(0, *feature_shape),
                maxshape=(None, *feature_shape),
                chunks=True,
                compression='gzip',
                compression_opts=9
            )

            label_dataset = label_file.create_dataset(
                'labels',
                shape=(0, *label_shape),
                maxshape=(None, *label_shape),
                chunks=True,
                compression='gzip',
                compression_opts=9
            )

            # Add first batch
            feature_dataset.resize(features.shape[0], axis=0)
            feature_dataset[:features.shape[0]] = features

            label_dataset.resize(labels.shape[0], axis=0)
            label_dataset[:labels.shape[0]] = labels

            current_size = features.shape[0]

        # Process remaining batches
        for images, labels in dataset.skip(1):
            features = base_model.predict(images)

            new_size = current_size + features.shape[0]

            feature_dataset.resize(new_size, axis=0)
            feature_dataset[current_size:new_size] = features

            label_dataset.resize(new_size, axis=0)
            label_dataset[current_size:new_size] = labels

            current_size = new_size


def load_features(feature_file, label_file):
    with h5py.File(feature_file, 'r') as f_features, h5py.File(label_file, 'r') as f_labels:
        return f_features['features'][:], f_labels['labels'][:]


class MetricsLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.metrics_data = []
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as file:
                try:
                    content = file.read().strip()
                    if content:
                        self.metrics_data = json.loads(content)
                except json.JSONDecodeError:
                    logging.info(f"Warning: {metrics_file} contains invalid JSON. Starting fresh.")

    def on_epoch_end(self, epoch, logs=None):
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': logs.get('loss'),
            'val_loss': logs.get('val_loss'),
            'train_accuracy': logs.get('accuracy'),
            'val_accuracy': logs.get('val_accuracy')
        }
        self.metrics_data.append(epoch_metrics)
        with open(metrics_file, 'w') as file:
            json.dump(self.metrics_data, file, indent=4)


if __name__ == '__main__':
    data_dir = 'datasets'
    model_dir = 'embedding_resnet50'
    os.makedirs(model_dir, exist_ok=True)
    metrics_file = os.path.join(model_dir, 'embedding_metrics_resnet50.json')

    indoor_dir = os.path.join(data_dir, 'indoor_processed')
    indoor_train = 'RotatedImages'
    indoor_test = 'RotatedTestImages'

    sun_dir = os.path.join(data_dir, 'SUN397_processed')
    sun_train = 'RotatedImages'
    sun_test = 'RotatedTestImages'

    indoor_train_dir = os.path.join(indoor_dir, indoor_train)
    indoor_test_dir = os.path.join(indoor_dir, indoor_test)

    sun_train_dir = os.path.join(sun_dir, sun_train)
    sun_test_dir = os.path.join(sun_dir, sun_test)

    # Load Indoor dataset
    indoor_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        indoor_train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset='training'
    )

    indoor_validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        indoor_train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset='validation'
    )

    indoor_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        indoor_test_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False
    )

    # Load SUN dataset
    sun_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        sun_train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset='training'
    )

    sun_validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        sun_train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset='validation'
    )

    sun_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        sun_test_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False
    )

    ### EMBEDDINGS
    # Concatenate datasets
    train_ds = indoor_train_ds.concatenate(sun_train_ds)
    validation_ds = indoor_validation_ds.concatenate(sun_validation_ds)
    test_ds = indoor_test_ds.concatenate(sun_test_ds)

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    base_model = ResNet50(weights='imagenet', include_top=False)

    # Process and save features batch by batch
    logging.info("Processing training data...")
    save_features_batch(
        train_ds,
        base_model,
        os.path.join(model_dir, 'train_features.h5'),
        os.path.join(model_dir, 'train_labels.h5'),
        BATCH_SIZE
    )

    logging.info("Processing validation data...")
    save_features_batch(
        validation_ds,
        base_model,
        os.path.join(model_dir, 'val_features.h5'),
        os.path.join(model_dir, 'val_labels.h5'),
        BATCH_SIZE
    )

    logging.info("Processing test data...")
    save_features_batch(
        test_ds,
        base_model,
        os.path.join(model_dir, 'test_features.h5'),
        os.path.join(model_dir, 'test_labels.h5'),
        BATCH_SIZE
    )

    # Load features for training
    logging.info("Loading features for training...")
    train_features, train_labels = load_features(
        os.path.join(model_dir, 'train_features.h5'),
        os.path.join(model_dir, 'train_labels.h5')
    )
    val_features, val_labels = load_features(
        os.path.join(model_dir, 'val_features.h5'),
        os.path.join(model_dir, 'val_labels.h5')
    )
    test_features, test_labels = load_features(
        os.path.join(model_dir, 'test_features.h5'),
        os.path.join(model_dir, 'test_labels.h5')
    )

    # Build and train model
    model = build_model(train_features.shape[1:], NUM_CLASSES)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint_path = os.path.join(model_dir, 'model_epoch_{epoch:02d}.keras')
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        save_best_only=False,
        save_weights_only=False,
        verbose=1
    )
    metrics_logger = MetricsLogger()

    # Train the model
    history = model.fit(
        train_features,
        train_labels,
        epochs=200,
        validation_data=(val_features, val_labels),
        callbacks=[early_stopping, model_checkpoint, metrics_logger]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_features, test_labels)
    logging.info(f"Test accuracy: {test_acc:.4f}")
    logging.info(f"Test loss: {test_loss:.4f}")
