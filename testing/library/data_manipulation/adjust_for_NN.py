import numpy as np


class shape_data():

    def __init__(self):
        pass

    def find_values(self):
        pass

    def set_values(self, cut=None:
        self.tresh_hold = cut

    def adjust_data_0_1(self, x, min=-1, max=1):
        data = x
        data = np.clip(data ,min ,max)
        data += min
        data /= (min+max)
        return data
