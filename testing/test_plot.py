import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import random as rr
plt.style.use('seaborn-darkgrid')
#%config InlineBackend.figure_format = 'svg'

snr = 4

mu = 0
sigma = np.sqrt(9/10**(snr/10)/2)

height_distr = stats.norm(loc=mu, scale=sigma)
r = np.arange(-10, 10, 0.001)
s = np.array([rr.randint(0,1)*2-1 for i in range(r.shape[0])])*3

spread = np.random.standard_normal(r.shape)
print(height_distr.pdf(0))

gamma = 10**(snr/10) #SNR to linear scale
if s.ndim==1:# if s is single dimensional vector
    P=1*sum(abs(s)**2)/len(s) #Actual power in the vector
else: # multi-dimensional signals like MFSK
    P=1*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
print(P)
N0=P/gamma # Find the noise spectral density
n = np.sqrt(N0/2)*np.random.standard_normal(s.shape) # computed noise
t = s + n # received signal
sigma= np.sqrt(N0/2)
height_distr2 = stats.norm(loc=mu, scale=sigma)

plt.plot(r, height_distr.pdf(r), color="r");
plt.plot(r, height_distr2.pdf(r), color="g");
plt.hist(n, bins=50, density=True, color="b");
plt.show()
