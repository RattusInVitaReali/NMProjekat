import keras
from keras.utils import image_dataset_from_directory
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

MAIN_PATH = './testing/'
IMG_SIZE = (200, 320)
BATCH_SIZE = 128
VALIDATION_SPLIT = 0
EPOCHS = 1000
PATIENCE = 10
LEARNING_RATE = 0.001
SEED = 69420

DATASET = image_dataset_from_directory(MAIN_PATH,
                                       image_size=IMG_SIZE,
                                       batch_size=BATCH_SIZE,
                                       crop_to_aspect_ratio=True,
                                       seed=SEED)

classes = DATASET.class_names

model = keras.models.load_model('./models')

labels = np.array([])
pred = np.array([])
for img, lab in DATASET:
    labels = np.append(labels, lab)
    prediction = np.argmax(model.predict(img, verbose=0), axis=1)
    pred = np.append(pred, prediction)
    # print(prediction)
    # print(lab)
    # for i in range(len(prediction)):
    #     if prediction[i] == lab[i]:
    #         plt.imshow(img[i].numpy().astype('uint8'))
    #         plt.title("CORRECT: " + classes[lab[i]] + " PREDICTED: " + classes[prediction[i]])
    #         plt.show()

print('ACCURACY : ' + str(100 * accuracy_score(labels, pred)) + '%')

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot(include_values=False)
plt.xticks(rotation=90)
plt.show()
