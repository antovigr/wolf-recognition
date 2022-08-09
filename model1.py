# Accuracy on test: 0.99
# Accuracy on validation: 0.96

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.image as img
import PIL.Image as Image
import cv2

import os
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
sns.set()

data_dir = "input"
data_dir = pathlib.Path(data_dir)

wolf = list(data_dir.glob('wolf/*'))
not_wolf = list(data_dir.glob('not_wolf/*'))

df_images = {
'wolf' : wolf,
'not_wolf' : not_wolf,
}

df_labels = {
'wolf' : 0,
'not_wolf' : 1,
}

images, labels = [], []
for label, pictures in df_images.items():
    for image in pictures:
        img = cv2.imread(str(image))
        if not(img is None):
            resized_img = cv2.resize(img, (224, 224)) #MobileNetv2 model
            images.append(resized_img)
            labels.append(df_labels[label])
images = np.array(images)
images = images/255
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

def setupModel():

    mobile_net = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
    mobile_net = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'
    mobile_net = hub.KerasLayer(
        mobile_net, input_shape=(224,224, 3), trainable=False)

    num_label = 2

    model = keras.Sequential([
    mobile_net,
    keras.layers.Dense(num_label)
    ])

    model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])


    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test,y_test))

    model.save("saved_model/model1.h5")

def getModel():
    return tf.keras.models.load_model('saved_model/model1')


# Accuracy graph

def accuracyGraph():

    model = getModel()

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test,y_test))

    plt.plot(history.history['acc'], color='b', label="Training accuracy")
    plt.plot(history.history['val_acc'], color='r',label="Validation accuracy")

    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


# Confusion matrix

def confusionMatrix():

    model = getModel()

    plt.clf()

    predict = model.predict(X_test)

    y_pred = np.argmax(predict, axis=-1)
    print(y_pred)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.show()



