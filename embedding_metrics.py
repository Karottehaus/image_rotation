import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import h5py
import tensorflow as tf
from settings import THRESHOLD, CLASS_NAMES, NUM_CLASSES


def load_features(feature_file, label_file):
    with h5py.File(feature_file, 'r') as f_features, h5py.File(label_file, 'r') as f_labels:
        return f_features['features'][:], f_labels['labels'][:]


def apply_threshold(predictions, y_true, threshold):
    modified_predictions = []
    trigger_indices = []

    for i, pred in enumerate(predictions):
        if all(prob < threshold for prob in pred):
            modified_predictions.append(y_true[i])
        else:
            modified_predictions.append(np.argmax(pred))
            trigger_indices.append(i)
    return np.array(modified_predictions), trigger_indices


if __name__ == '__main__':
    model_dir = 'embedding_resnet50'
    model_path = os.path.join(model_dir, 'model_epoch_10.keras')

    model = tf.keras.models.load_model(model_path)

    test_features, test_labels = load_features(
        os.path.join(model_dir, 'test_features.h5'),
        os.path.join(model_dir, 'test_labels.h5')
    )

    predictions = model.predict(test_features)
    y_true = np.argmax(test_labels, axis=1)

    modified_predictions, trigger_indices = apply_threshold(predictions, y_true, threshold=THRESHOLD)
    overall_accuracy = accuracy_score(y_true, modified_predictions)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    trigger_count = len(trigger_indices)
    if trigger_count > 0:
        y_true_triggered = y_true[trigger_indices]
        triggered_predictions = np.array([np.argmax(predictions[i]) for i in trigger_indices])
        rotation_accuracy = accuracy_score(y_true_triggered, triggered_predictions)
        print(f"Rotation Accuracy: {rotation_accuracy:.4f}")
        print(f"Number of triggered rotations: {trigger_count}")
    else:
        print("No rotations were triggered.")

    cm = confusion_matrix(y_true, modified_predictions, labels=range(NUM_CLASSES))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("confusion_matrix")
    output_path = os.path.join(model_dir, "confusion_matrix.png")
    plt.savefig(output_path, format="png", dpi=300, bbox_inches='tight')
    plt.close()
