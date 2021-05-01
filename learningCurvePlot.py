# Short file to implement the learning curve plots

import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, manifold
os.environ['KMP_DUPLICATE_LIB_OK']='True'

numSamples = np.array([1/16, 1/8, 1/4, 1/2]) * 51000
trainAcc = [99.81, 99.84, 99.80, 99.77]
testAcc = [97.45, 98.36, 98.75, 98.84]

plt.figure()
plt.loglog(numSamples, trainAcc, label='Train')
plt.loglog(numSamples, testAcc, label='Test')
plt.xlabel('(Log) Number of Training Samples Used')
plt.ylabel('(Log) Accuracy of Trained Model')
plt.title('Log-Log Learning Curve')
plt.legend()
