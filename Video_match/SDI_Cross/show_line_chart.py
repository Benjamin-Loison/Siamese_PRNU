import matplotlib.pyplot as plt
import numpy as np
#
# x = [3,5,7,9,11,13,20,30,40,50,60,70,80,90,100]
# y_proposed = [97.26,96.28,95.50,95.01,94.81,88.94,95.11,93.15,93.74,88.26,87.77,84.83,85.91,76.13,72.70]
# y_pce = [87.67,87.38,87.00,87.18,87.18,86.59,83.07,80.14,76.52,74.17,69.37,65.66,59.30,53.82,47.16]
# y_pcn = [52.54,46.67,41.19,38.94,37.97,34.44,40.90,38.55,47.95,36.50,35.52,37.00,41.09,40.70,41.88]

x = [10,20,30,40,50,60,70,80,90,100]
y_proposed = [94.81,95.11,93.15,93.74,88.26,87.77,84.83,85.91,76.13,72.70]
y_pce = [87.18,83.07,80.14,76.52,74.17,69.37,65.66,59.30,53.82,47.16]
y_pcn = [37.97,40.90,38.55,47.95,36.50,35.52,37.00,41.09,40.70,41.88]
plt.plot(x,y_proposed,label='Proposed',marker='o')
plt.plot(x,y_pce,label='PCE',marker='o')
plt.plot(x,y_pcn,label='PCN',marker='o')
for i in range(len(x)):

    plt.text(x[i],y_proposed[i],y_proposed[i],label='Proposed',ha='center',va='bottom')
    plt.text(x[i],y_pce[i],y_pce[i],label='PCE',ha='center',va='top')
    plt.text(x[i],y_pcn[i],y_pcn[i],label='PCN',ha='center',va='bottom')
#加上线的属性指标,loc则使属性放在哪个位置,framealpha为透明度
# plt.yticks(range(0, 110, 10))  # y轴的刻度
plt.ylabel('Accuracy (%)')
plt.xlabel('Shift Pixel (pixel)')
plt.xticks(x,x)
plt.grid(axis='x', ls='--')
# plt.title('Comparison of matching accuracy of different models in anti-shift experiments')
plt.legend()
plt.show()
