import tensorflow as tf
import tensorflow.keras as tfk
from tabulate import tabulate
import zipfile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from datetime import datetime
import tensorflow_hub as hub
import itertools
from sklearn.metrics import confusion_matrix
import pandas as pd


class Khelper:
    def __init__(self, train_dir, test_dir, val_dir=None):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.class_name = os.listdir(self.train_dir)
        self.log_dirs = {}
    
    # Image Preprocessing

    def image_dataset_from_dir(self, image_size=(224, 224), label_mode='categorical', batch_size=32, shuffle=True, seed=None):
        
        """
        Return image from train_data, test_data and validation_data 
        preprocessed by `TensorFlow.Keras.preprocessing.image_dataset_from_directory()`
        
        args:
            train_dir (str):
            test_dir (str):
            val_dir (str): 
            image_size (tuple): Default (224, 224) dimension of the output image height and image width 
            label_mode (str): Default 'categorical' https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
            batch_size (int): Default 32 quantity of batch passed during each epoch
            shuffle (Bool): Default True shuffling the data or not passing False
            seed (int): Random Seed 
            
        outputs:
            if val_dir is passed then return
                train_data, test_data, val_data
            otherwise:
                train_data, test_data
        """
        # train data
        train_data = tfk.preprocessing.image_dataset_from_directory(
            directory=self.train_dir,
            image_size=image_size,
            label_mode=label_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed
        )
    
        # test data
        test_data = tfk.preprocessing.image_dataset_from_directory(
            directory=self.test_dir,
            image_size=image_size,
            label_mode=label_mode,
            batch_size=batch_size,
            shuffle=False,
            seed=seed
        )
        
        if self.val_dir:
            # validation data
            val_data = tfk.preprocessing.image_dataset_from_directory(
                directory=self.val_dir,
                image_size=image_size,
                label_mode=label_mode,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed
            )
            
            return train_data, test_data, val_data
        
        return train_data, test_data
    
    @staticmethod
    def create_model_from_url(model_url, input_shape=(224, 224, 3), num_classses=10, trainable=False):
        """
        takes tensorflow hub URL and create a
        sequential model.
        Return the model to be compiled.
        """
        # download pretrained model
        feature_extractor_layer = hub.KerasLayer(model_url,
                                                trainable=trainable,
                                                name="feature_extraction_layer",
                                                # (224, 224, 3)
                                                input_shape=input_shape)

        # Create the model
        model = tfk.Sequential([
            feature_extractor_layer,
            tfk.layers.Dense(num_classses,
                            activation='softmax',
                            name='output_layer')
        ])

        return model


    def walk_through_dir(self, dir_path):
        """
        Walks through dir_path and prints a summary of its contents.
        
        Args:
            dir_path (str): Target directory path.
        """
        data = []

        # Esamina tutte le sottodirectory e i file nella dir_path
        for root, dirs, files in os.walk(dir_path):
            # Ignora le sottodirectory nascoste
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            # Conta il numero di immagini nella sottodirectory corrente
            images_count = len([f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))])

            # Aggiungi i dati della sottodirectory alla lista
            data.append([os.path.basename(root), len(dirs), images_count, root])

        # Definisci gli header per la tabella
        table_headers = ['Directory Name', 'Number of Subdirectories', 'Number of Images', 'Directory Path']

        # Stampare la tabella utilizzando tabulate
        print(tabulate(data, headers=table_headers, tablefmt="fancy_grid"))
        
    @staticmethod    
    def unzip_data(filename):
        """
        Unzips filename into the current working directory
        Args:
            filename (str): a filepath to a target zip folder to be unzipped.
        """
        zip_ref = zipfile.ZipFile(filename, "r")
        zip_ref.extractall()
        zip_ref.close()
        
        
    def create_tensorboard_callback(self, dir_name, experiment_name, verbose=0):
        """
        Creates a TensorBoard callback instand to store log files
        Stores log files with the filepath: 'dir_name/experiment_name/current_datetime/'
        Args:
            dir_name: target directory to store TensorBoard log files
            experiment_name: name of experiment directory (e.g. model_1)
            verbose: 0 or 1. if setted to 1 print on screen destination folder where TensorBoard log files where stored.
        """
        log_dir = dir_name + "/" + experiment_name + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dirs[experiment_name] = log_dir
        tensorboard_callback = tfk.callbacks.TensorBoard(
            log_dir=log_dir, 
            histogram_freq=1  # Enable histogram computation for every epoch.
            )
        
        if verbose == 1:
            print(f"Saving TensorBoard log files to: {log_dir}")
            
        return tensorboard_callback
    

    def image_data_augmentation(self, 
                                input_shape, 
                                rescaling=False, 
                                RandomFlip="horizontal", 
                                RandomRotation=0.2, 
                                RandomHeight=0.2, 
                                RandomWidth=0.2,
                                RandomBrightness=None,
                                RandomContrast=None):
        # Define input layer with specified shape
        inputs = tfk.layers.Input(shape=input_shape)
        
        # Create data augmentation pipeline
        x = inputs
        
        if rescaling:
            x = tfk.layers.Rescaling(1./255)(x)
        
        x = tfk.layers.RandomFlip(RandomFlip)(x)
        
        # RandomBrightness check
        if RandomBrightness and rescaling:
            x = tf.keras.layers.RandomBrightness(RandomBrightness, value_range=(0, 1))(x)
        elif RandomBrightness:
            x = tf.keras.layers.RandomBrightness(RandomBrightness, value_range=(0, 255))(x)
            
        # RandomContrast check   
        if RandomContrast:
            x = tf.keras.layers.RandomContrast(RandomContrast)(x)
        
        x = tfk.layers.RandomRotation(factor=RandomRotation)(x)
        x = tfk.layers.RandomHeight(factor=RandomHeight)(x)
        x = tfk.layers.RandomWidth(factor=RandomWidth)(x)
        
        # Model for data augmentation
        data_augmentation_model = tfk.models.Model(inputs=inputs, outputs=x, name="data_augmentation_model")
        
        return data_augmentation_model

    
    @staticmethod
    def get_model_from_url(model_url, img_shape=(224, 224, 3), trainable=False):
        """
        takes tensorflow hub URL and create a
        sequential model
        """
        # download pretrained model
        feature_extractor_layer = hub.KerasLayer(model_url,
                                                 trainable=trainable,
                                                 name="feature_extraction_layer",
                                                 input_shape=img_shape)
            
        return feature_extractor_layer
    
    
    def plot_random_image_from_train_dir(self, verbose=0, label=False):
        """
        Plot a random Image from dir_path and return the image array.
        """
        # Get class name
        class_names = os.listdir(self.train_dir)
        random_dir = np.random.randint(0, len(class_names))
        
        # all images paths given the random dir
        images = os.listdir(os.path.join(self.train_dir, class_names[random_dir]))
        
        # random image
        rand_img = np.random.randint(0, len(images))
        img_path = os.path.join(self.train_dir, class_names[random_dir], images[rand_img])
        img = Image.open(img_path)
        
        # plot
        if verbose != 0:    
            plt.title(class_names[random_dir])
            plt.imshow(img)
            plt.axis(False)
            plt.show()
    
        if label:
            return (np.array(img), class_names[random_dir])
        
        return np.array(img)
    
    
    def get_picture_info_text(self, array_pic):
        """
        Return text containing information about the image array.
        """
        # Decide se l'immagine è già scalata
        scaled_info = "is Scaled" if array_pic.max() <= 1 else "isn't already Scaled"
    
        # Decidi il formato dell'immagine
        if array_pic.ndim == 3:
            format_info = "in RGB format"
        elif array_pic.ndim == 4:
            format_info = "in RGBA format"
        else:
            format_info = f"in an unrecognized format (ndim={array_pic.ndim})"
    
        # Costruisci e restituisci il testo delle informazioni sull'immagine
        table = [
            ["Scaled Status", scaled_info],
            ["Image Format", format_info],
            ["Shape", str(array_pic.shape)],
            ["Value Range", f"({array_pic.min()}, {array_pic.max()})"],
            ["Data Type", str(array_pic.dtype)],
            ["Total Number of Elements", array_pic.size]
        ]

    # Formatta la tabella usando tabulate
        info_table = tabulate(table, headers=["Attribute", "Value"], tablefmt="fancy_grid")

        return info_table
    
    
    def load_and_scale_data(filename, img_shape=224):
        """
        Reads an image from filename,
        turn into tensor and reshape it.
        """
        # read in the image
        img = tf.io.read_file(filename)
    
        # Decode the file into tensor
        img = tf.image.decode_image(img)
    
        # Resize the image
        img = tf.image.resize(img, size=[img_shape, img_shape])
    
        # Rescale image with values between 0 and 1
        img = img/255.
    
        return img
    
    
    def pred_and_plot(self, compiled_model, filename):
        """
        Import a single image from a direct path,
        make a prediction and plot it with information.
        """

        # Import target image and preprocess it
        img = self.load_and_scale_data(filename)
    
        # Make prediction
        pred = compiled_model.predict(tf.expand_dims(img, axis=0), verbose=0)
    
        # Get the predoction class
        pred_class = self.class_names[int(tf.round(pred))]
    
        # Plot image and pred class
        plt.imshow(img)
        plt.title(f"Predicted Label: {pred_class} \nwith Accuracy of: {np.round(pred[0][0]*100, 2)}%")
        plt.axis(False);
        
        
    def multiple_pred_and_plot(self, compiled_model, n_plot=5):
        """
        Perform multiple predictions using the compiled model and plot the results in subplots.
        """
        # Dynamic subplot grid
        num_rows = (n_plot + 2) // 3  # number of required rows
        num_cols = min(3, n_plot)  # Maximum 3 columns per row
    
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
        axes = axes.flatten()  # Flatten the array to facilitate iteration
    
        for i in range(n_plot):
            class_name = np.random.choice(self.class_names)
            dir_path = os.path.join(self.test_dir, class_name)
    
            images = os.listdir(dir_path)
            selected_images = np.random.choice(images, n_plot, replace=False)
            images_path = [os.path.join(dir_path, img) for img in selected_images]
    
            filename = images_path[i]
            img = self.load_and_scale_data(filename)
            pred = compiled_model.predict(tf.expand_dims(img, axis=0), verbose=0)
            pred_class = compiled_model.class_names[int(tf.round(pred))]
    
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(f"Predicted Label: {pred_class}\nAccuracy: {np.round(pred[0][0]*100, 2)}%")
            ax.axis(False)
    
        plt.tight_layout()
        plt.show()
        
        
    def make_confusion_matrix(self, y_true, y_pred, figsize=(10, 10), text_size=15):
        # create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalization
        n_classes = cm.shape[0]
    
        # prettify
        fig, ax = plt.subplots(figsize=figsize)
        # Create matrix plot
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
    
        # Set labels to be classes
        if self.class_name:
            labels=self.class_name
        else:
            labels = np.arange(cm.shape[0])
    
        # Label the axes
        ax.set(title='Confusion Matrix',
                xlabel='Predict Label',
                ylabel='True Label',
                yticks=np.arange(n_classes),
                xticks=np.arange(n_classes),
                xticklabels = labels,
                yticklabels = labels)
    
        # Set x-axis labels to bottom
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()
    
        # Adjust label size
        ax.yaxis.label.set_size(text_size)
        ax.xaxis.label.set_size(text_size)
        ax.title.set_size(text_size)
    
        # set treshold for different color
        treshold = (cm.max() + cm.min()) / 2.
    
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                    horizontalalignment='center',
                    color="white" if cm[i, j] > treshold else 'black',
                    size=text_size)
    
    
    
    def plot_model_history(self, history, figsize=(12, 5)):
        plt.figure(figsize=figsize)  # Imposta la dimensione della figura

        # accuracies plot
        plt.subplot(1, 2, 1)  # first subplot
        pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot(grid=True, ax=plt.gca())
        plt.title('Accuracy')  
        plt.xlabel('Epochs')  
        plt.ylabel('Accuracy') 

        # losses plot
        plt.subplot(1, 2, 2)  # second subplot
        pd.DataFrame(history.history)[['loss', 'val_loss']].plot(grid=True, ax=plt.gca())
        plt.title('Loss')  
        plt.xlabel('Epochs')  
        plt.ylabel('Loss')  

        plt.tight_layout() 
        plt.show()