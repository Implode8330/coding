import scipy.stats
import random as r
import numpy    as np
import math     as m

class Turbo_Encoder(object):

    def __init__(self, encrypt=None):
        if encrypt !=None:
            self.set_encryption(encrypt)

        self.encoding = self.__encoding_structure_convolution

    def set_encryption(self, D):
        if type(D) == list:
            D = np.array(D)
        self.coding = D

    def get_bitlength(self):
        return 1

    def get_code(self):
        return self.coding

    def __encoding_structure_convolution(self,x):
        conv_code = []
        code_l = self.coding.shape[0]
        if type(x)==list:
            shape = len(x)
        else:
            shape = x.shape[0]
        op_code   = np.zeros(shape*code_l, dtype=np.int8)
        for d in self.coding:
            conv_code.append( np.convolve(d, x) )
        for i in range(shape):
            for j,c in enumerate(conv_code):
                op_code[i*code_l+j] = c[i]
        return op_code%2

    def __encoding_structure_delay(self,x):
        pass


    def encode(self, x):
        code = self.encoding(x)
        return code



class BCJR_Decoder():

    def __init__(self, version=None):
        if version == "BSC":
            self.LLR_Ybk = self.LLR_Ybk_bsc
        elif version == "AWGN":
            self.LLR_Ybk = self.LLR_Ybk_awgn


    def set_variables(self, **kwarg):
        if "distribution" in kwarg.keys():
            self.distribution = kwarg["distribution"]
        if "encoding_structure" in kwarg.keys():
            e = self.format_encode(kwarg["encoding_structure"])
            self.encoding_structure = e
        if "message_length" in kwarg.keys():
            self.K = kwarg["message_length"]
        if "version" in kwarg.keys():
            if kwarg["version"] == "BSC":
                self.LLR_Ybk = self.LLR_Ybk_bsc
                self.__log_gamma = self.__log_gamma2
            elif kwarg["version"] == "AWGN":
                self.LLR_Ybk = self.LLR_Ybk_awgn
                self.__log_gamma = self.__log_gamma1
        if "channel_info" in kwarg.keys():
            amp   = kwarg["channel_info"][0]
            sigma = kwarg["channel_info"][1]
            self.CHANNEL_CONST = 2*amp/sigma

        self.preprocess()

    def format_encode(self, c):
        l = []
        for row in c:
            l.append(len(row))
        max_l   = max(l)
        cc      = np.zeros( (len(c), max_l), dtype=np.int16 )
        for i in range(len(c)):
            cc[i][:l[i]] = c[i]
        return cc


    def preprocess(self):
        self.check_info()
        self.extracting_extra_variables()

    def check_info(self):
        #raise "Nothing checked"
        pass

    def extracting_extra_variables(self):
        self.count_states()
        self.graph_transitions()

    def graph_transitions(self):
        self.next_M = np.zeros(( self.n_states, 2, 2), dtype=np.int16)
        tran_M = np.zeros(( self.n_states, self.n_states ,self.K), dtype=np.int16)
        state_bits = [2**i for i in range(self.n_states)]
        possible_s = [0]
        steal      = False
        for s in range(self.n_states):
            next_s_b0, y_b0 = self.next_state(0, s)
            next_s_b1, y_b1 = self.next_state(1, s)
            self.next_M[s,:,:] = [ [next_s_b0, y_b0], [next_s_b1, y_b1] ]
        for k in range(self.K):
            coming_states = []
            for s in possible_s:
                next_s_b0, y_b0 = self.next_state(0, s)
                next_s_b1, y_b1 = self.next_state(1, s)
                tran_M[s,[next_s_b0, next_s_b1], k] = [1,1]
                coming_states.append( next_s_b0 )
                coming_states.append( next_s_b1 )
            possible_s = np.unique( coming_states )
        possible_s = [0]
        for k in range(self.K-1, -1, -1):
            not_pos_s  = [i for i in range(self.n_states) if i not in possible_s ]
            tran_M[:,not_pos_s,k] = 0
            possible_s = np.where( tran_M[:,possible_s,k] == 1 )[0]
            if possible_s.shape[0] == self.n_states:
                break
        self.tran_M = tran_M

    def next_state(self, b, s):
        bit_s       = self.convert_to_bits(s)
        bit_next_s  = np.r_[b,bit_s]
        next_s      = self.convert_to_num(bit_next_s[:-1])
        x = self.encoding_structure.dot(bit_next_s)%2
        return next_s, x[1:]



    def convert_to_bits(self, n):
        return n//self.bit_conv %2

    def convert_to_num(self, b):
        return b.dot(self.bit_conv)

    def count_states(self):
        code = self.encoding_structure
        states = 0
        for c in code:
            temp = c.shape[0]-1
            if temp < 0 : raise
            if temp > states : states = temp
        self.n_states   = 2**states
        self.bit_states = states
        if states < 128:
            set_type = np.int16
        else:
            set_type = np.int16
        self.bit_conv   = np.array([2**i for i in range(self.bit_states)], dtype=set_type)

    def calculate_graph(self):
        self.calc_log_gamma_matrix()
        # print("gamma is nan/inf : ", np.any(np.isnan(self.gamma_M) ), np.any(np.isinf(self.gamma_M) ) )
        self.calc_log_alpha_matrix()
        # print("alpha is nan/inf : ", np.any(np.isnan(self.alpha_M) ), np.any(np.isinf(self.alpha_M) ) )
        self.calc_log_beta_matrix()
        # print("beta  is nan/inf : ", np.any(np.isnan(self.beta_M) ), np.any(np.isinf(self.beta_M) ) )
        self.calc_log_prob_matrix()
        # print("prob  is nan/inf : ", np.any(np.isnan(self.prob_M) ), np.any(np.isinf(self.prob_M) ) )
        self.calc_logLikelihood_B_of_y()
        # print("LLRB  is nan/inf : ", np.any(np.isnan(self.llr_B) ), np.any(np.isinf(self.llr_B) ) )

    def calc_log_gamma_matrix(self):
        Y = self.input_Y
        self.gamma_M = np.zeros((self.n_states, self.n_states, self.K))
        for k in range(self.K):
            for s_prim in range(self.n_states):
                for s in range(self.n_states):
                    self.gamma_M[s_prim, s, k] = \
                                self.__log_gamma(Y, k, s_prim, s)

    def __log_gamma1(self, Y, k, s_prim, s):
        yb, yx  = Y[k]
        sb      = self.convert_to_bits(s)
        b       = sb[0]
        ss, x   = self.next_state(b, s_prim)
        if not self.tran_M[s_prim,s,k]:
            return -np.inf  #Not possible transition
        LYb     = self.LLR_Ybk(yb, b, k)
        LYx     = self.LLR_Ybk(yx, x, k)
        gamma   = (LYx + LYb)/2
        return gamma

    def LLR_Ybk_awgn(self,y,b,k):
        return y*((-1)**b)

    def __log_gamma2(self, Y, k, s_prim, s):
        if not self.tran_M[s_prim,s,k]:
            return -np.inf  #Not possible transition
        yb, yx  = Y[k]
        b       = s%2
        x       = self.next_M[s_prim,b,1]
        Lb      = self.LLR_Bk(b,k)
        LYb     = self.LLR_Ybk(yb, b, k)
        LYx     = self.LLR_Ybk(yx, x, k)
        # gamma   = (1-2*b)*(Lb + LYb)/2  + (1-2*x)*LYx/2
        gamma   = ((-1)**b)*LYb + ((-1)**x)*LYx
        gamma   /= 2
        # if s_prim == 3 and s in [2,3] and k in [1,2]:
        #     print(f"b {b, LYb}")
        #     print(f"x {x, LYx}")
        #     print(f"s {gamma}")
        return gamma
    #
    # def __log_gamma3(self, Y, k, s_prim, s):
    #     yb, yx  = Y[k]
    #     sb      = self.convert_to_bits(s)
    #     b       = sb[0]
    #     ss, x   = self.next_state(b, s_prim)
    #     if not self.tran_M[s_prim,s,k]:
    #         return -np.inf  #Not possible transition
    #     Lb      = self.LLR_Bk(b,k)
    #     LYb     = self.LLR_Ybk(yb, b, k)
    #     LYx     = self.LLR_Ybk(yx, x, k)
    #     gamma   = (1-2*b)*(Lb + LYb)/2  + (1-2*x)*LYx/2
    #     return gamma


    def LLR_Bk(self,b,k):
        return 0

    def LLR_Ybk_bsc(self,y,b,k):
        ##insert some log ratio to decibel
        return 3.9*((-1)**y)


    def calc_log_alpha_matrix(self):
        self.alpha_M = np.zeros((self.n_states, self.K))
        for k in range(self.K):
            for s in range(self.n_states):
                prev_max = None
                if k == 0:
                    self.alpha_M[:,k] = [0 if i == 0 else -np.inf for i in range(self.n_states)]
                    continue
                for s_prim in range(self.n_states):
                    this_addition = self.alpha_M[s_prim,k-1] + self.gamma_M[s_prim,s,k-1]
                    prev_max = self.max_star(this_addition, prev_max)
                self.alpha_M[s,k] = prev_max


    def max_star(self, a, b):
        if b == None or b == -np.inf:
            return a
        if a == -np.inf:
            return b
        return max(a,b) + m.log(1+ m.exp(-abs(a-b)))

    def calc_log_beta_matrix(self):
        self.beta_M = np.zeros((self.n_states, self.K+1))
        for k in range(self.K, -1, -1):
            for s_prim in range(self.gamma_M.shape[-2]):
                prev_max = None
                if k == self.K:
                    self.beta_M[:,k] = [0 if i == 0 else -np.inf for i in range(self.n_states)]
                    continue
                for s in range(self.n_states):
                    this_addition = self.beta_M[s,k+1] + self.gamma_M[s_prim,s,k]
                    prev_max = self.max_star(this_addition, prev_max)
                self.beta_M[s_prim,k] = prev_max

    def calc_log_prob_matrix(self):
        self.prob_M = np.zeros((self.n_states, self.n_states , self.K))
        for k in range(self.K):
            alpha = 0
            for s_prim in range(self.n_states):
                for s in range(self.n_states):
                    self.prob_M[s_prim,s,k] = \
                                self.beta_M[s,k+1] \
                            +   self.gamma_M[s_prim,s,k]\
                            +   self.alpha_M[s_prim,k]

                    self.printt(s_prim,s,k)

    def printt(self,s_prim,s,k):
        return
        print("######################")
        print("s' s k")
        print(s_prim,s,k)
        print("a",self.alpha_M[s_prim,k])
        print("g",self.gamma_M[s_prim,s,k])
        print("b",self.beta_M[s,k])
        print("p",self.prob_M[s_prim,s,k])
        print("######################")

    def calc_logLikelihood_B_of_y(self):
        self.llr_B = np.zeros((self.K))
        self.llr_B_ = np.zeros((self.K,2))
        for k in range(self.K):
            max_1 = max_0 = None
            for s_prim in range(self.n_states):
                for s in range(self.n_states):
                    if s%2:         #message passed is a 0 or 1. if true, its odd, which first bit is 1.
                        max_1 = self.max_star(self.prob_M[s_prim, s, k], max_1)
                    else:
                        max_0 = self.max_star(self.prob_M[s_prim, s, k], max_0)

            self.llr_B_[k] = [max_0, max_1]
            self.llr_B[k]  = max_0 - max_1

    def decode(self, message):
        self.input_Y = np.zeros((self.K, 2))
        l_x = 1
        for k in range(self.K) :
            idx = k*(l_x+1)
            b = message[idx]
            x = message[idx+1:idx+1+l_x]
            self.input_Y[k] = [b,x]
        self.calculate_graph()
        bit_guess = (np.sign(self.llr_B)-1)*-1/2
        return bit_guess


################################################################
#####shiiiitcode################################################
################################################################
#####shiiiitcode################################################
################################################################
#####shiiiitcode################################################
################################################################
#####shiiiitcode################################################
################################################################
#####shiiiitcode################################################
################################################################
#####shiiiitcode################################################
################################################################
#####shiiiitcode################################################
################################################################
#####shiiiitcode################################################
################################################################
#####shiiiitcode################################################
################################################################
#####shiiiitcode################################################
################################################################
np.set_printoptions(precision=1)
np.random.seed(0)
r.seed(0)

if __name__ == "__main__":
    from transmission_characteristics import *

    enc     = [[1],[1,1,1]]
    amp     = 1
    sigm    = 1
    u_length= 10
    tb      = Turbo_Encoder(enc)
    tc      = Transmission_Channel(amp, sigm)
    d_bsc   = BCJR_Decoder()
    d_bsc.set_variables( encoding_structure =enc,
                        message_length     = u_length,
                        # version            = "AWGN",
                        version            = "BSC",
                        channel_info       = [amp, sigm])
    d_awgn  = BCJR_Decoder()
    d_awgn.set_variables( encoding_structure =enc,
                        message_length     = u_length,
                        version            = "AWGN",
                        # version            = "BSC",
                        channel_info       = [amp, sigm])
    checks = 100
    sa = sb = sc  = 0
    ssa = ssb = ssc = 0
    for i in range(checks):
        d = np.array([[r.getrandbits(u_length-2)]]) #last 2 are 0 state reseters
        u = (((d[0,None] & (1 << np.arange(u_length)))) > 0).astype(np.int8)[0]
        x = tb.encode(u)
        ya = tc.AWGN(x)
        yb = (np.sign(ya)-1)*-1/2
        g_a = d_awgn.decode(ya)
        g_b = d_bsc.decode(yb.astype(np.int16))
        sa += np.sum(g_a==u)/len(u)
        sb += np.sum(g_b==u)/len(u)

        if np.all(g_a==u):
            ssa += 1
        elif np.all(g_b==u):
            print(d_awgn.llr_B)
            print(d_bsc.llr_B)
        if np.all(g_b==u):
            ssb += 1
        # else:
        #     print(f"errors {np.sum(y!=x)}")
        #     print(y.astype(np.int))
        #     print(x)
        #     print(u)
        #     print(g.astype(np.int))
        #     print(bcjr.llr_B)
    sa /= checks
    sb /= checks
    ssa /= checks
    ssb /= checks
    print(f"BER  a{sa:.3f} b{sb:.3f} c{sc:.3f}")
    print(f"BLER a{ssa:.3f} b{ssb:.3f} c{ssc:.3f}")

if __name__ == "__main__2":
    enc     = [[1],[1,1,1]]
    u       = [1,0,1,1,0,0]
    tb      = Turbo_Encoder(enc)
    bcjr    = BCJR_Decoder()
    bcjr.set_variables( encoding_structure =enc,
                        message_length  = len(u))
    y = tb.encode(u)
    s = 0
    for i in range(len(y)):
        yy = y.copy()
        yy[i] = 0
        g = bcjr.decode(yy)
        s += np.sum(g==u)/len(u)
        if np.sum(g==u)/len(u) < 1:
            print(bcjr.llr_B,g,u)
    # print(bcjr.tran_M)
    s /= len(y)
    print("BER", s)

if __name__ == "__main__3":
    enc = [[1],[1,1,1]]
    bcjr = BCJR_Decoder()
    bcjr.set_variables( encoding_structure=enc,
                        message_length=8)
    bcjr.next_state(0,3)

if __name__ == "__main__2":
    enc = [[1,1,1],[1,0,1]]
    tb  = Turbo_Encoder(enc)
    u = [1,1,0,1,0,0,0,0]
    y = tb.encode(u)
    bitlength = tb.get_bitlength()
    print(y)
