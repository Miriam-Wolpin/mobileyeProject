import pickle

from TFL_manager import TflManager
import numpy as np


class Controller:
    """
    supposed to take a pls and and then perform each module of the TFL manager on the png of a specific frame
    """

    def __init__(self, file_path):
        with open(file_path) as the_file:
            paths = the_file.read().split("\n")
        self.frames = paths[1:]
        self.pkl = paths[0]

    def calc_em(self, i):
        with open(self.pkl, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        return data['egomotion_' + str(i - 1) + '-' + str(i)]

    def run(self):
        tfl_manager = TflManager(self.pkl)
        for index, frame in enumerate(self.frames):
            print(f"*** {index + 1} ***")
            candidates1, colors1 = tfl_manager.part1(frame)
            candidates2, auxiliary2 = tfl_manager.part2(frame, candidates1, colors1)

            if index:
                em = self.calc_em(index + 24)
                tfl_manager.part3(self.frames[index - 1], frame, prev[0], candidates2, prev[1], auxiliary2, em,
                                  index + 24)
            prev = (candidates2, auxiliary2)
