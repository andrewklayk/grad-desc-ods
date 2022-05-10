import matplotlib.pyplot as plt
import numpy as np

time_hist_gd = np.loadtxt('time_hist_gd.txt')
acc_hist_gd = np.loadtxt('acc_hist_gd.txt')
time_hist_bcr = np.loadtxt('time_hist_bcgdr.txt')
acc_hist_bcr = np.loadtxt('acc_hist_bcgdr.txt')
time_hist_bcc = np.loadtxt('time_hist_bcgdc.txt')
acc_hist_bcc = np.loadtxt('acc_hist_bcgdc.txt')

_, ax = plt.subplots(nrows=1)
ax.plot(time_hist_gd, acc_hist_gd)
ax.plot(time_hist_bcc, acc_hist_bcc)
ax.plot(time_hist_bcr, acc_hist_bcr)
ax.set_ylabel('Accuracy')
ax.set_xlabel('CPU time')
ax.legend(['GD', 'Cyclic BCGD', 'Randomized BCGD'])

plt.savefig('acc_time_plots')