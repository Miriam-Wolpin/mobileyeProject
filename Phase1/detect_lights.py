try:
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image
    import matplotlib.pyplot as plt
except ImportError:
    raise
def convolve(img):
    kernel = np.ones((3, 3)) / -9
    kernel[1, 1] = 8 / 9
    return ndimage.convolve(img, kernel)
def get_red_light_positions(c_image):
    convolved_red = convolve(c_image)
    red_filtered_img = ndimage.maximum_filter(convolved_red, 10)
    red_position = np.argwhere(red_filtered_img == convolved_red)
    red_position = list(filter(lambda l: convolved_red[l[0]][l[1]] > 0.1, red_position))
    x_red = [p[0] for p in red_position]
    y_red = [p[1] for p in red_position]
    return y_red, x_red
def get_green_light_positions(c_image):
    convolved_green = convolve(c_image)
    green_filtered_img = ndimage.maximum_filter(convolved_green, 10)
    green_position = np.argwhere(green_filtered_img == convolved_green)
    green_position = list(filter(lambda l: convolved_green[l[0]][l[1]] > 0.1, green_position))
    x_green = [p[0] for p in green_position]
    y_green = [p[1] for p in green_position]
    return y_green, x_green
def find_lights(c_image, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    print(c_image)
    c_image = plt.imread(c_image).copy()
    x_red, y_red = get_red_light_positions(c_image[:, :, 0])
    x_green, y_green = get_green_light_positions(c_image[:, :, 1])
    light_candidte = []
    candidate_color = []
    for x, y in zip(x_red, y_red):
        light_candidte.append([x, y])
        candidate_color.append('r')
    for x, y in zip(x_green, y_green):
        light_candidte.append([x, y])
        candidate_color.append('g')
    plt.imshow(c_image)
    plt.plot(x_red, y_red, 'r.')
    plt.imshow(c_image)
    plt.plot(x_green, y_green, 'g.')
    plt.show()
    return light_candidte, candidate_color