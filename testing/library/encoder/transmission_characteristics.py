import numpy as np
import scipy.stats

class Transmission_Channel():

    def __init__(self, sign_str=None, variance=None):
        if sign_str != None:
            self.A   = sign_str
        if variance != None:
            self.sigma = variance
        print()
        print()
        cdf = scipy.stats.norm(0,self.sigma).cdf(self.A)
        print(f"Amp {sign_str}, Var{variance}, Error Rate: {1-cdf:.5f}")


    def AWGN(self, x):
        b = (-1)**x
        w = np.random.normal(0, self.sigma, x.shape)
        return self.A*b+w
