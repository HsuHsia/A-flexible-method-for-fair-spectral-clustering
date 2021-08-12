import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

plt.figure()

axis_x = np.arange(5, 21, 1)
x_major_locator = MultipleLocator(5)

f3 = [0.160162841,0.13396393,0.107780138,0.098917821,0.090343332,0.090429523,0.07721432,0.069726341,0.065918319,0.06115924,0.057094579,0.053924315,0.051431668,0.048350082,0.046009466,0.043811488]
f1 = [0.140601942,0.111755403,0.089120753,0.092573115,0.089568428,0.081177767,0.069743715,0.063320675,0.05811783,0.054862838,0.053258351,0.051014865,0.045282213,0.046062156,0.041422663,0.039982105]
f2 = [0.158695089,0.134363453,0.100788244,0.102384889,0.089568428,0.090866026,0.076755397,0.068582799,0.063459769,0.059394015,0.055909129,0.053538132,0.051139637,0.048531588,0.046586976,0.044604638]
plt.rc('font',family='Times New Roman')
plt.subplot(1,4,1)
plt.plot(axis_x, f1, label='UFSC', color="r", linestyle="--")
plt.plot(axis_x, f2, label='SCFC', color="b", linestyle=":")
plt.plot(axis_x, f3, label='SC', color="g", linestyle="-.")
plt.xlabel('cluster numbers', fontsize=13)
plt.ylabel('AED', fontsize=13)
plt.yticks(size = 13)
plt.xticks(size = 13)
plt.title('Obesity(gender)', fontsize=18)
plt.legend(fontsize=16)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

f6 = [0.10784993,0.082602909,0.071935968,0.060257083,0.049683843,0.042882111,0.037279502,0.03395442,0.032557257,0.033812034,0.033012862,0.031347219,0.028274376,0.026731062,0.024961112,0.023897775]
f4 = [0.070814223,0.060167312,0.051772386,0.04557225,0.036127292,0.035631067,0.031376783,0.030290018,0.028295328,0.029589478,0.026669537,0.024511897,0.024249312,0.022156979,0.021330945,0.020983763]
f5 = [0.10784993,0.082246946,0.068193628,0.053766528,0.044787083,0.039858741,0.033423469,0.0319789,0.032799315,0.03431442,0.031338023,0.029451704,0.026342631,0.02479438,0.023194707,0.022191815]

plt.subplot(1,4,2)
plt.plot(axis_x, f4, label='UFSC', color="r", linestyle="--")
plt.plot(axis_x, f5, label='SCFC', color="b", linestyle=":")
plt.plot(axis_x, f6, label='SC', color="g", linestyle="-.")
plt.xlabel('cluster numbers',fontsize=13)
plt.ylabel('AED', fontsize=13)
plt.yticks(size = 13)
plt.xticks(size = 13)
plt.title('Obesity(family_history)', fontsize=18)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

f9 = [0.109406259,0.080425131,0.068874873,0.082492155,0.074052341,0.071069828,0.073616162,0.072198787,0.066106285,0.052019617,0.050503108,0.053166944,0.048158872,0.043519505,0.043181006,0.03972491]
f7 = [0.105212675,0.083105965,0.063198746,0.055798555,0.052328802,0.045894746,0.042795251,0.046060524,0.041453705,0.037964604,0.03239644,0.03598849,0.034823055,0.034006107,0.030112016,0.030930183]
f8 = [0.179831111,0.10815671,0.102025204,0.08722391,0.081498924,0.081599173,0.0707878,0.060584731,0.056872479,0.055975892,0.043715404,0.042234756,0.042592369,0.040067736,0.04417754,0.036472615]


plt.subplot(1,4,3)
plt.plot(axis_x, f7, label='UFSC', color="r", linestyle="--")
plt.plot(axis_x, f8, label='SCFC', color="b", linestyle=":")
plt.plot(axis_x, f9, label='SC', color="g", linestyle="-.")
plt.xlabel('cluster numbers',fontsize=13)
plt.ylabel('AED', fontsize=13)
plt.yticks(size = 13)
plt.xticks(size = 13)
plt.title('Bank(marital)', fontsize=18)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

f12 = [0.00546634,0.004743467,0.004180328,0.008923097,0.007469683,0.006390103,0.006388515,0.005603765,0.002423681,0.00446033,0.004037043,0.002001573,0.003378705,0.001792705,0.002892387,0.002695745]
f10 = [0.005448209,0.004730799,0.004194443,0.003743201,0.003385293,0.003084807,0.002826271,0.002615816,0.002430044,0.00226836,0.002128162,0.002003423,0.001893282,0.001792459,0.001702489,0.001622795]
f11 = [0.005450968,0.004730799,0.008636611,0.006584585,0.007143612,0.006581021,0.005903506,0.003517063,0.004782897,0.00429717,0.003989965,0.003639486,0.003175836,0.002869118,0.00280105,0.00256918]

plt.subplot(1,4,4)
plt.plot(axis_x, f10, label='UFSC', color="r", linestyle="--")
plt.plot(axis_x, f11, label='SCFC', color="b", linestyle=":")
plt.plot(axis_x, f12, label='SC', color="g", linestyle="-.")
plt.xlabel('cluster numbers',fontsize=13)
plt.ylabel('AED', fontsize=13)
plt.yticks(size = 13)
plt.xticks(size = 13)
plt.title('Bank(default)', fontsize=18)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
#
# print('Spectral Clustering results :', np.mean(unfair_results))
# print('Fair Spectral Clustering results :', np.mean(fair_results))
# plt.xlabel('k')
# # Set the y axis label of the current axis.
# plt.ylabel('Balance')
# # Set a title of the current axes.
# # show a legend on the plot
# Display a figure.
plt.show()