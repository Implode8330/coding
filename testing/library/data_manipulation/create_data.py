from library.encoder.transmission_characteristics   import Transmission_Channel
from library.decoder.BCJR   import BCJR_Decoder, Turbo_Encoder
import random   as r
import numpy    as np
import pickle   as pkl
import os
from tqdm import tqdm

class data_generator():

    def __init__(self, length, SNRdb, seed=None):
        self.length = length
        self.SNRdb = SNRdb
        self.path = f"/home/amydos/data/SNRdb_{SNRdb}_{length*2}_bit/"
        # self.path = "/home/amydos/Dropbox/Exjobb/coding/testing/data/amp_1_var_1_32_bit/"
        if seed != None:
            r.seed(seed)

    def generate_64_bit(self ):
        self.generate_u_bit(32)
    def generate_32_bit(self ):
        self.generate_u_bit(16)
    def generate_20_bit(self ):
        self.generate_u_bit(10)
    def generate_16_bit(self ):
        self.generate_u_bit(8)
    def generate_u_bit(self, u_length):
        enc     = [[1],[1,1,1]]
        amp     = 1
        sigm    = np.sqrt(amp**2/10**(self.SNRdb/10)/2)
        samples = 10000
        tb      = Turbo_Encoder(enc)
        tc      = Transmission_Channel(amp, sigm)
        bcjr = BCJR_Decoder()
        bcjr.set_variables( encoding_structure = enc,
                            message_length     = u_length,
                            version            = "AWGN",
                            # version            = "BSC",
                            channel_info       = [amp, sigm])
        temp     = self.generate_info_bits(samples)
        values   = temp[1]
        corr_y   = temp[0]
        pred_y   = np.zeros(corr_y.shape)
        data_x   = np.zeros((samples,u_length*2))
        print(data_x.shape)
        for i in tqdm(range(samples)):
            x  = tb.encode(corr_y[i])
            ya = tc.AWGN(x)
            g_a = bcjr.decode(ya)
            data_x[i] = ya
            pred_y[i] = bcjr.llr_B
        self.save_sample_set(values, "values_")
        self.save_sample_set(data_x, "data_x_")
        self.save_sample_set(corr_y, "data_y_")
        self.save_sample_set(pred_y, "pred_y_")

    def save_sample_set(self, data, filename):
        num = self.next_number( filename)
        str = f"{self.path}{filename}{num}.pkl"
        pkl.dump(data, open( str, "wb" ))

    def next_number(self, filename):
        n = 0
        while os.path.exists(f"{self.path}{filename}{n}.pkl"):
            n += 1
        return n

    def collect_previous_bits(self):
        n=0
        already_used = np.zeros((1,1))
        print("HWH",os.path.exists(f"{self.path}values_{n}.pkl"), f"{self.path}values_{n}.pkl")
        while os.path.exists(f"{self.path}values_{n}.pkl"):
            with open(f"{self.path}values_{n}.pkl" ,"rb") as file:
                data = pkl.load(file)
                if already_used.shape == (1,1):
                    already_used = data
                else:
                    already_used = np.r_[already_used, data]
                print(already_used.shape)
            n+=1
        return already_used

    def generate_info_bits(self, sample_size=1):
         #last 2 are 0 state reseters
        previous_used = self.collect_previous_bits()
        d = []
        while len(d) != sample_size:
            v = r.getrandbits(self.length-2)
            if not np.any(np.isin(v, previous_used) ) or self.length < 16 :
                if [v] not in d:
                    d.append([v])
            else:
                if len(d) > 0 :
                    pass
                    # print(f"THere is a {v} in {d, previous_used} previous_used : {np.any(np.isin(v, previous_used) )}")
        print("klar")
        d = np.array(d)
        u = (((d[:] & (1 << np.arange(self.length)))) > 0).astype(np.int)
        return u, d.squeeze(1)

    def get_data(self, index=0):
        with open(f"{self.path}data_x_{index}.pkl", "rb") as file:
            x = pkl.load(file)
        with open(f"{self.path}data_y_{index}.pkl", "rb") as file:
            y = pkl.load(file)
        with open(f"{self.path}pred_y_{index}.pkl", "rb") as file:
            yy= pkl.load(file)
        with open(f"{self.path}values_{index}.pkl", "rb") as file:
            v = pkl.load(file)
        return (x,y,yy), v

    def mini_load_function(self, idx):
        load,v = self.get_data(idx)
        print(f"Loaded {self.path[-20:]}data_y_{idx}.pkl")
        data_x, corr_y, pred_y = load
        clip_lvl = 5
        pred_y = -1*pred_y
        pred_y = pred_y.clip(-1*clip_lvl,clip_lvl) + clip_lvl
        pred_y /= clip_lvl*2
        return (data_x, corr_y, pred_y), v


def accuracy_measuring(pred , corr):
    if np.any(pred < 0):
        pred[pred < 0] = 1
        pred[pred > 1] = 0
    if not np.any(pred < 0):
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
    accuracy = np.sum(pred == corr)/pred.shape[0]/pred.shape[1]
    print(f" The Accuracy is # {accuracy} #")



if __name__ == "__main__":
    code_gen = data_generator(16, seed = 0)
    corr_y   = code_gen.generate_32_bit()
