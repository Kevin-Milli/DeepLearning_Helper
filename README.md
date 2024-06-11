# Khelper: TensorFlow Keras Helper

Khelper is a utility class designed to streamline common tasks in deep learning workflows using TensorFlow and Keras. It provides various methods for data preprocessing, model creation, visualization, and evaluation.

## Installation

To use Khelper, simply copy the `khelper.py` file into your project directory and import it in your Python scripts as follows:

```python
from khelper import Khelper
```

## Usage

Initializing Khelper
You can initialize Khelper by providing the paths to your training and testing directories:

```python
helper = Khelper(train_dir, test_dir, val_dir=None)
```

## Image Preprocessing

`image_dataset_from_dir()`
This method creates image datasets from directories using TensorFlow's `image_dataset_from_directory()` function. It returns preprocessed datasets for training, testing, and validation (if provided).

```python
train_data, test_data, val_data = helper.image_dataset_from_dir(image_size=(224, 224), label_mode='categorical', batch_size=32, shuffle=True, seed=None)
```

## Model Creation

`create_model_from_url()`
Creates a sequential model from a TensorFlow Hub URL.

```python
model = helper.create_model_from_url(model_url, input_shape=(224, 224, 3), num_classses=10, trainable=False)
```

## Data Augmentation

`image_data_augmentation()`
Creates a data augmentation model using TensorFlow data augmentation layers.

```python
data_augmentation_model = helper.image_data_augmentation(input_shape, rescaling=False, RandomFlip="horizontal", RandomRotation=0.2, RandomHeight=0.2, RandomWidth=0.2, RandomBrightness=None, RandomContrast=None)
```

## Model Training Visualization
`plot_model_history()`
Plots the training history of a machine learning model including accuracy and loss over epochs.

```python
helper.plot_model_history(history, figsize=(12, 5))
```

## Evaluation
`make_confusion_matrix()`
Generates and visualizes a confusion matrix for evaluating the performance of a classification model.

```python
helper.make_confusion_matrix(y_true, y_pred, figsize=(10, 10), text_size=15)
```
