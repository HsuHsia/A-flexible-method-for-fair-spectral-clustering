import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

plt.figure()

axis_x = np.arange(0, 11000, 1000)
x_major_locator = MultipleLocator(1000)

f1 = [0,0.50527643,2.165474081,6.429221869,14.81721487,27.74951544,47.04882026,74.01225181,102.8363706,142.4761129,192.8829067]
f2 = [0,0.250143385,1.208938742,3.66299901,8.511608028,16.01863647,27.24450216,43.09287386,60.6751111,84.14956231,113.8227474]
f3 = [0,0.220986664,0.842451954,2.629132652,6.089063454,11.31328878,19.26613598,29.92209563,40.93870549,55.70418062,76.734971]

f4 = [0,0.223267984,1.214106297,3.938525009,9.019451737,17.10261631,28.60583434,46.06570113,64.87254808,88.77021506,122.9859651]
f5 = [0,0.183167744,0.851790929,2.836690497,6.370004225,12.03455555,20.25121398,31.53758972,44.19062634,60.21199379,83.80394211]
plt.rc('font',family='Times New Roman')
plt.subplot(1,2,1)
plt.plot(axis_x, f1, label='UFSC', color="r", linestyle="--")
plt.plot(axis_x, f2, label='SCFC', color="b", linestyle=":")
plt.plot(axis_x, f3, label='SC', color="g", linestyle="-.")
plt.xlabel('point numbers', fontsize=16)
plt.ylabel('Running Time [s]', fontsize=14)
plt.yticks(size = 14)
plt.xticks(size = 14)
# plt.title('Running Time', fontsize=18)
plt.legend(fontsize=16)
ax = plt.gca()


plt.subplot(1,2,2)
plt.plot(axis_x, f4, label='MFSC', color="r", linestyle="--")

plt.plot(axis_x, f5, label='SC', color="g", linestyle="-.")
plt.xlabel('point numbers', fontsize=16)
plt.ylabel('Running Time [s]', fontsize=14)
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.show()
ax.xaxis.set_major_locator(x_major_locator)