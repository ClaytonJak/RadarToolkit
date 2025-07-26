import numpy as np
import matplotlib as mpl
import wgn

# generate an array for signal
size_test_sig = 100
test_sig = np.zeros(size_test_sig)
test_sig[int(size_test_sig/2)] = 1

Y = wgn.awgn(test_sig,1)
print(Y)


