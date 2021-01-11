import pickle
import numpy as np


class PKL:
    def __init__(self, path):
        with open(path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        self.focal = data['flx']
        self.pp = data['principle_point']
        self.EM = np.eye(4)
