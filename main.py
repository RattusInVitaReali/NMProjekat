import logging
import math

import numpy as np
import tensorflow as tf
from keras import mixed_precision
from keras import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Hvala bratu s Stack Overflowa
def create_class_weight(labels_dict, mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


tf.get_logger().setLevel(logging.ERROR)

# GPU magic
mixed_precision.set_global_policy('mixed_float16')


MAIN_PATH = './archive/'
IMG_SIZE = (200, 320)
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.25
EPOCHS = 1000
PATIENCE = 10
LEARNING_RATE = 0.001
SEED = 69420

Xtrain = image_dataset_from_directory(MAIN_PATH,
                                      subset='training',
                                      validation_split=VALIDATION_SPLIT,
                                      image_size=IMG_SIZE,
                                      batch_size=BATCH_SIZE,
                                      crop_to_aspect_ratio=True,
                                      seed=SEED)

Xval = image_dataset_from_directory(MAIN_PATH,
                                    subset='validation',
                                    validation_split=VALIDATION_SPLIT,
                                    image_size=IMG_SIZE,
                                    batch_size=BATCH_SIZE,
                                    crop_to_aspect_ratio=True,
                                    seed=SEED)

classes = Xtrain.class_names
print(classes)

count = np.zeros(len(classes), dtype=np.int32)
for _, labels in Xtrain:
    y, _, c = tf.unique_with_counts(labels)
    count[y.numpy()] += c.numpy()

print(count)

total_count = np.sum(count)

counts = {}

for i in range(len(classes)):
    counts[i] = count[i]

print(counts)

weights = create_class_weight(counts)

print(weights)

num_classes = len(classes)

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.Conv2D(64, 5, strides=(2, 2), padding='same', activation='relu'),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.25),
    layers.Dense(256, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax', dtype='float32')
])
model.summary()

model.compile(Adam(learning_rate=LEARNING_RATE),
              loss=SparseCategoricalCrossentropy(),
              metrics='accuracy')

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=PATIENCE, verbose=1)

history = model.fit(Xtrain,
                    epochs=EPOCHS,
                    validation_data=Xval,
                    class_weight=weights,
                    callbacks=[es],
                    verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

labels = np.array([])
pred = np.array([])
for img, lab in Xval:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

print('ACCURACY : ' + str(100 * accuracy_score(labels, pred)) + '%')

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot(include_values=False)
plt.xticks(rotation=90)
plt.show()

model.save("./models")
