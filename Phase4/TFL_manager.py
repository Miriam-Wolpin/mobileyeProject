import pickle
import numpy as np

from part2 import detect_tfls
from Phase1.detect_lights import find_lights
from Phase3.SFM_standAlone import FrameContainer, visualize
from Phase3 import SFM


class TflManager:
    def __init__(self, path):
        with open(path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        self.focal = data['flx']
        self.pp = data['principle_point']

    def part1(self, img):
        candidates, colors = find_lights(img)
        return candidates, colors

    def part2(self, img, candidates, auxiliary):
        # detect_tfls(img, candidates)
        return [[1116, 330], [932, 171], [536, 335]], ['r', 'r', 'g']

    def part3(self, prev_img, curr_img, prev_tfls, curr_tfls, prev_aux, curr_aux, em, id):
        curr_container = FrameContainer(curr_img)
        curr_container.traffic_light = np.array(curr_tfls)
        curr_container.EM = em
        prev_container = FrameContainer(prev_img)
        prev_container.traffic_light = np.array(prev_tfls)

        curr_container = SFM.calc_TFL_dist(prev_container, curr_container, self.focal, self.pp)
        visualize(prev_container, curr_container, self.focal, self.pp, id)
