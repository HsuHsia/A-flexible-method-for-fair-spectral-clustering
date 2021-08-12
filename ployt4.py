import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

plt.figure()

axis_x = np.arange(5, 21, 1)
x_major_locator = MultipleLocator(5)
f1 = [0.262970788,0.225702135,0.197475993,0.181216633,0.166931567,0.152429817,0.150132716,0.136893939,0.125807376,0.118305164,0.109940354,0.105735161,0.10025648,0.09404932,0.088898244,0.084102826]
f2 = [0.262970788,0.225702135,0.177475993,0.165121663,0.146931567,0.150705107,0.140466552,0.123027722,0.121807376,0.113305164,0.0979940354,0.095735161,0.093338851,0.09404932,0.088898244,0.084102826]

plt.rc('font',family='Times New Roman')
plt.subplot(2,2,1)
plt.plot(axis_x, f2, label='MFSC', color="r", linestyle="--")
plt.plot(axis_x, f1, label='SC', color="g", linestyle=":")
plt.xlabel('cluster numbers', fontsize=13)
plt.ylabel('AED', fontsize=13)
plt.yticks(size = 13)
plt.xticks(size = 13)
plt.title('Drug(gender, ethnicity)', fontsize=18)
plt.legend(fontsize=16)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

f3 = [0.229576577,0.192818199,0.178941262,0.163627761,0.157700801,0.149337649,0.130291177,0.117816092,0.104548716,0.094310834,0.087825249,0.080313104,0.07536206,0.07401122,0.071060855,0.067132421]
f4 = [0.161595352,0.172590731,0.171026338,0.157912245,0.152783325,0.135450912,0.115308864,0.102228778,0.094202139,0.085304262,0.081240434,0.079996476,0.077167153,0.075638467,0.072175046,0.067274975]

plt.subplot(2,2,2)
plt.plot(axis_x, f4, label='MFSC', color="r", linestyle="--")
plt.plot(axis_x, f3, label='SC', color="g", linestyle=":")
plt.xlabel('cluster numbers',fontsize=13)
plt.ylabel('AED', fontsize=13)
plt.yticks(size = 13)
plt.xticks(size = 13)
plt.title('Adult(sex, income)', fontsize=18)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)


f5 = [0.269890633,0.165765215,0.173456707,0.13495819,0.134746749,0.12346299,0.121565574,0.131272582,0.103067696,0.096941435,0.084848203,0.079886514,0.073777588,0.068130511,0.07186427,0.070028594]
f6 = [0.225512471,0.159645594,0.157163335,0.118245819,0.141006138,0.116327887,0.11470647,0.11754156,0.106953783,0.094791375,0.07300338,0.070805612,0.065379451,0.065929702,0.067635961,0.068652464]

plt.subplot(2,2,3)
plt.plot(axis_x, f6, label='MFSC', color="r", linestyle="--")

plt.plot(axis_x, f5, label='SC', color="g", linestyle=":")
plt.xlabel('cluster numbers',fontsize=13)
plt.ylabel('AED', fontsize=13)
plt.yticks(size = 13)
plt.xticks(size = 13)
plt.title('Obesity(gender, family_history)', fontsize=18)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
#
f7 = [0.103129724,0.088858208,0.078694563,0.090844907,0.080620827,0.081030566,0.077651525,0.072774083,0.063673238,0.058368811,0.053774577,0.05118923,0.049926053,0.047123963,0.045558433,0.041445335]
f8 = [0.103129724,0.088858208,0.078694563,0.079529265,0.071806294,0.065454208,0.06383299,0.056097009,0.051098982,0.050509141,0.048654977,0.047935054,0.046016107,0.042060216,0.039960993,0.036617217]

plt.subplot(2,2,4)
plt.plot(axis_x, f8, label='MFSC', color="r", linestyle="--")
plt.plot(axis_x, f7, label='SC', color="g", linestyle=":")
plt.xlabel('cluster numbers',fontsize=13)
plt.ylabel('AED', fontsize=13)
plt.yticks(size = 13)
plt.xticks(size = 13)
plt.title('Bank(marital, default)', fontsize=18)
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
