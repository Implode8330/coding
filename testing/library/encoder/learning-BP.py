import numpy as np
from library.encoder import LDPC


class encode_base(object):
    def __init__(self, matrix):
        pass




class Belief_Propagation(object):

    def __init__(self, **kwarg):
        arg_keys = kwarg.keys()
        if "encoder_matrix" in arg_keys:
            if "convert_function" not in arg_keys:
                raise "en matrix kom med men ingen convert function"
                f = kwarg["convert_function"]
                decode_M = f(kwarg["encoer_matrix"])
        elif "decoder_matrix" in arg_keys :
            decode_M = kwarg["decoder_matrix"]
        else:
            raise "nu kom de inte med någon matrix"
        self.M       = decode_M
        self.M_shape = decode_M.shape
        self.h = self.h_parsing(decode_M)

        if "epsilon" in arg_keys:
            self.set_epsilon( kwarg["epsilon"] )
        else:
            self.epsilon = None

    def h_parsing(self, M):
        h_functions = []
        for h in M:
            h_functions.append( np.where(h == 1)[0] )
        return h_functions

    def set_epsilon(self, e):
        self.epsilon = e

    def LLR(self, data):
        return (-1)**data * np.log((1-self.epsilon)/self.epsilon)

    def init_data(self, data):
        x_LLR = self.LLR(data)
        self.u0 = np.multiply(self.M, x_LLR)
        self.first_iter = True


    def left_iter(self):
        if self.first_iter:
            self.v = self.u0.copy()
            self.first_iter = False
            return
        self.v  = self.u0.copy()
        u_q     = np.sum(self.c, axis=0)-self.c
        for h_p in range(self.M_shape[0]):
            for u in range(self.M_shape[1]):
                if self.M[h_p,u] == 0: continue
                self.v[h_p, u] += u_q[h_p,u]

    def right_iter(self):
        self.c = np.zeros(self.M_shape)
        v_q    = np.tanh(self.v/2)
        for h in range(self.M_shape[0]):
            for u_p in range(self.M_shape[1]):
                if self.M[h,u_p] == 0: continue
                u_q = np.delete(self.h[h], np.where(self.h[h] == u_p) )
                temp = np.prod(v_q[h,u_q])
                self.c[h,u_p] = 2*np.arctanh(temp)

    def node_likelihood(self):
        self.u0 = self.u0+np.multiply(self.M, np.mean(self.c,axis=0) )

    def iteration(self):
        self.left_iter()
        self.right_iter()
        self.node_likelihood()


    def predict(self, x, iter=1):
        if self.epsilon == None: raise "no epsilon selected"

        self.init_data(data)
        for i in range(iter):
            self.iteration()

    def bit_prediction(self):
        d = np.mean(self.u0, axis=0)
        return ((np.sign(d)-1)/-2).astype(int)

def read_matrixfile(name):
    m = np.read_csv(name)
    return m

if __name__ == "__main__":

    m = np.array( [ [1, 0, 0, 1, 1, 0, 1],
                    [1, 0, 0, 0, 1, 0, 1],
                    [0, 1, 0, 1, 1, 1, 1],
                    [0, 1, 0, 0, 1, 1, 1],
                    [0, 0, 1, 0, 0, 1, 1],
                    [0, 0, 1, 1, 0, 1, 1],])
    encoder = m.T
    bp = Belief_Propagation(decoder_matrix=m, epsilon=0.25)
    count = correct = 0
    print("MMMMM\n",m)
    print("MMMMM\n",np.m)
    for num in range(2**0):
        bits= num//np.array([2**5,2**4,2**3,2**2,2**1,2**0])%2
        data = encoder.dot(bits)%2
        print("#################################")
        print("bits\n",bits)
        print("data\n",data)
        print("#################################")
        for i in range(data.shape[0]):
            corrupt_bits    = data.copy()
            corrupt_bits[i] = (data[i]+1)%2
            bp.predict(corrupt_bits, iter=5)
            out = bp.bit_prediction()
            if np.all(out == data):
                print("correct")
                correct += 1
            else:
                check = m.dot(corrupt_bits)%2
                if np.all(check == np.zeros(m.shape[0])):
                    print("correct")
                    correct += 1
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\ncheck",check)
                else:
                    print("check",check)
                    print("failed")
            print("out :", out)
            print("data:", data)
            print("corr:", corrupt_bits)

            count += 1
    print(correct/count)
        # print("In ", data.astype(int))
        # print("Out", out)




        # np.set_printoptions(precision=5)
