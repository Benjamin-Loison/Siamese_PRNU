import matplotlib.pyplot as plt
import numpy as np

x = [3,5,7,9,11,13]
y_proposed = [97.26,96.28,95.50,95.01,94.81,88.94]
y_pce = [87.67,87.38,87.00,87.18,87.18,86.59]
y_pcn = [52.54,46.67,41.19,38.94,37.97,34.44]
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
plt.title('Comparison of matching accuracy of different models in anti-shift experiments')
plt.legend()
plt.show()
