import PIL
import cv2
import numpy as np
import scipy
from PIL import Image
from scipy import signal as sg
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max



def find(img):
    print()
    image = np.array(Image.open(img))
    print(image)


# find("data/scaleup_imgs/bremen_000116_000019_gtFine_color.png")

def convolve(mat):
    # im = Image.open("C:/Users/USER/Desktop/שושי/שנה ב/סמסטר ב/bootcamp/פרוייקטים/mobileye/project/kernel.png")
    # im.thumbnail((7, 7), Image.ANTIALIAS)
    # print(im.format, im.size, im.mode)
    # plt.imshow(im)
    # plt.show()

    # image = np.array(im)
    # image = image[:, 1: 6]
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3)) / -9
    kernel[1, 1] = 8 / 9
    # kernel = [[0, 0, 0], [0, 8/9, 0], [0, 0, 0]]
    # new_im = Image.fromarray(kernel)
    # new_im = new_im.resize((9, 9), PIL.Image.ANTIALIAS)

    return sg.convolve2d(mat, kernel)


# scipy.ndimage.maximum_filter


def max_filter(img):
    # peaks = peak_local_max(img, min_distance=5)
    peaks = scipy.ndimage.maximum_filter(img, size=10)
    peaks = np.argwhere(peaks == img)
    peaks = list(filter(lambda l: img[l[0]][l[1]] > 0.1, peaks))
    x_list = [p[0] for p in peaks]
    y_list = [p[1] for p in peaks]
    return x_list, y_list
    # return peaks



