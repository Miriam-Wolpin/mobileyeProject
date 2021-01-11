from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def read_images(img, labeled_img):
    img = plt.imread(img) * 255
    img = img.astype('uint8')
    labeled = np.asarray(Image.open(labeled_img))

    return img, labeled


def crop(x, y, img):
    return img[x - 81 // 2: x + 81 // 2 + 1, y - 81 // 2: y + 81 // 2 + 1, :]


def get_tfl_image(labeled_img, img):
    tfl_coordinates_arr = np.argwhere(labeled_img == 19)

    if len(tfl_coordinates_arr) == 0:
        return None

    pos = np.random.randint(len(tfl_coordinates_arr))
    pixels_fit_image = tfl_coordinates_arr[pos][0] < 50 or tfl_coordinates_arr[pos][0] > 900 or \
                       tfl_coordinates_arr[pos][1] < 50 or tfl_coordinates_arr[pos][1] > 2000

    if pixels_fit_image:
        return None

    return crop(tfl_coordinates_arr[pos][0], tfl_coordinates_arr[pos][1], img)


def get_random_image(img):
    for i in range(15):
        pos = np.random.randint(low=50, high=900)
        random_img = crop(pos, pos, img)
        if len(np.argwhere(19 == random_img)) == 0:
            return random_img
    return None


def crop_images(image, labeled_image, data_set_arr, labeled_arr):
    img, labeled_img = read_images(image, labeled_image)
    for i in range(2):
        tfl_image = get_tfl_image(labeled_img, img)
        if tfl_image is not None:
            data_set_arr.append(tfl_image)
            labeled_arr.append(1)
            data_set_arr.append(tfl_image[:, ::-1, :])
            labeled_arr.append(1)

        random_image = get_random_image(img)
        if random_image is not None:
            data_set_arr.append(random_image)
            labeled_arr.append(0)
            data_set_arr.append(random_image)
            labeled_arr.append(0)
    else:
        return


def read_dirs(dir):
    cities_dir = r"C:\Users\RENT\Desktop\mobileye\חומר\gtFine_trainvaltest\gtFine\{}".format(dir)
    cities = os.listdir(cities_dir)
    lable = []
    imgs = []
    for city in cities:
        print(city)
        labels_dir = rf"C:\Users\RENT\Desktop\mobileye\חומר\gtFine_trainvaltest\gtFine\{dir}\{city}\*gtFine_labelIds.png"

        img_dir = rf"C:\Users\RENT\Desktop\mobileye\חומר\leftImg8bit_trainvaltest\leftImg8bit\{dir}\{city}\*.png"

        lable += glob.glob(os.path.join(labels_dir))
        imgs += glob.glob(os.path.join(img_dir))

    return imgs, lable


def main(dir):
    data_set_arr = []
    labeled_arr = []
    print("before read_dir")
    img_arr, labeled_img_arr = read_dirs(dir)

    i = 1
    for img, labeled in zip(img_arr, labeled_img_arr):
        print(f'Processing{i}/{len(img_arr)}')
        i += 1
        crop_images(img, labeled, data_set_arr, labeled_arr)

    np.array(labeled_arr, dtype='uint8').tofile(f'data_dir/{dir}/labels.bin')
    np.array(data_set_arr, dtype='uint8').tofile(f'data_dir/{dir}/data.bin')


def show_image_and_label():
    labels_in_words = {0: "No TFL", 1: "TFL"}

    images = np.fromfile('C:/Users/RENT/Desktop/mobileye/project/Phase2/data_dir/train/data.bin', dtype='uint8')
    num_of_imgs = len(images) // (81 * 81 * 3)
    images = images.reshape(num_of_imgs, 81, 81, 3)

    labels = np.fromfile('C:/Users/RENT/Desktop/mobileye/project/Phase2/data_dir/train/labels.bin', dtype='uint8')

    plt.figure(figsize=(10, 10))
    for i in range(25):
        pos = np.random.randint(num_of_imgs)
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[pos], cmap=plt.cm.binary)
        plt.xlabel(labels_in_words[labels[pos]])
    plt.show()


def test_dataset():
    show_image_and_label()


if __name__ == "__main__":

    #main("train")
    #main("val")

    #test_dataset()
