import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from tensorflow import keras
from keras import layers
from keras.utils import image_dataset_from_directory

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 24

#Using the Xception model for fine-tuning
base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(80, 80, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.

base_model.trainable = False

#Display stats after the training is over
def display(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()
    
#Evaluate model
def test(test_data):
    test_dir = 'Dataset/ExpandedSet/test/Weeb'
    
    best_model = keras.models.load_model("Logs/weeb_finder.keras")
    for i in os.listdir(test_dir):
        img_path = fr'{test_dir}\{i}'
        print(img_path)

        img = cv2.imread(img_path, 3)
        # cv2.imshow('test', img)
        # cv2.waitKey(0)

        img = cv2.resize(img, (80, 80))
        img = np.reshape(img, [1, 80, 80, 3])

        pred = best_model.predict(img)
        print(pred)
        
    best_model.evaluate(test_data)

#Data augmentation block to prevent overfitting from little data
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
    ]
)

#Datasets made here
new_base_dir = r"Dataset/ExpandedSet"
train_dataset = image_dataset_from_directory(
    new_base_dir + r"/train",
    label_mode='categorical',
    image_size=(80, 80),
    batch_size=BATCH_SIZE)
validation_dataset = image_dataset_from_directory(
    new_base_dir + "/validation",
    label_mode='categorical',
    image_size=(80, 80),
    batch_size=BATCH_SIZE)
test_dataset = image_dataset_from_directory(
    new_base_dir + "/test",
    label_mode='categorical',
    image_size=(80, 80),
    batch_size=BATCH_SIZE)

#Building architecture here
inputs = keras.Input(shape=(80, 80, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = base_model(x)
x = layers.Flatten()(x)
x = layers.Dense(128)(x)
x = layers.Dense(64)(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'Adam',
    metrics = ['accuracy']
)

#Checkpoints made here
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = "Logs/weeb_finder.keras",
        save_best_only = True,
        monitor = "val_loss"
    ),
]

#Start training
history = model.fit(
    train_dataset,
    epochs = 50,
    validation_data = validation_dataset,
    callbacks = callbacks
)

#Show stats and then evaluate model on test set (weeb only)
display(history)
test(test_dataset)