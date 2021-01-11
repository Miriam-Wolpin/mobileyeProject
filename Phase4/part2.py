import tensorflow.keras.models as models
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt


def crop(x, y, img):
    return img[x - 81 // 2: x + 81 // 2 + 1, y - 81 // 2: y + 81 // 2 + 1, :]


def detect_tfls(img, pts):
    img = plt.imread(img) * 255
    img = img.astype('uint8')
    loaded_model = models.load_model("model.h5")
    imgs = np.array([crop(p[0], p[1], img) for p in pts])
    l_predictions = loaded_model.predict(imgs)
    sbn.distplot(l_predictions[:, 0]);

    l_predicted_label = np.argmax(l_predictions, axis=-1)
