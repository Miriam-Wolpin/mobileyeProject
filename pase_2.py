import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def display_data(index):
    # print(np.fromfile('labels.bin', dtype='uint8')[index])
    # print(np.array(np.fromfile('data.bin', dtype='uint8')).shape)
    plt.imshow(np.array(np.fromfile('data.bin', dtype='uint8').reshape(81,81,3)))

def crop(x, y, img):
    return img[x - 81 // 2:x + 81 // 2 + 1, y - 81 // 2:y + 81 // 2 + 1, :]

def insert_to_file(arr, file):
    np.array(arr).tofile(file)



lable = np.asarray(Image.open("bochum_000000_000313_gtFine_labelIds.png"))

img = plt.imread("bochum_000000_000313_leftImg8bit.png")
lable = lable.astype(np.uint8)
plt.imshow(lable,cmap = 'gray')

args = np.argwhere(lable == 19)
plt.imshow(crop(args[len(args) // 2][0], args[len(args) // 2][1], img))

img = img * 255
img = img.astype(np.uint8)
insert_to_file(crop(args[len(args) // 2][0], args[len(args) // 2][1], img), "data.bin")
insert_to_file([1], "labels.bin")
display_data(0)