import matplotlib.pyplot as plt
# 移位情况下的数据
# fusion_data = [0.935,0.793,0.693,0.606,0.5141]
# dwt_data = [0.779,0.544,0.435,0.354,0.272]
# spncnn_data = [0.895,0.703,0.589,0.487,0.402]
# dude_data = [0.89,0.706,0.609,0.512,0.42]
# fusion_rep = [0.943,0.907,0.882,0.816,0.786]
# dwt_rep = [0.85,0.752,0.709,0.599,0.643]
#
fusion_data = [0.93,0.79,0.69,0.60,0.51]
dwt_data = [0.77,0.54,0.43,0.35,0.27]
spncnn_data = [0.89,0.70,0.58,0.48,0.40]
dude_data = [0.89,0.70,0.60,0.51,0.42]
fusion_rep = [0.94,0.90,0.88,0.81,0.78]
dwt_rep = [0.85,0.75,0.70,0.59,0.64]
#裁剪情况下的数据
# fusion_data = [0.93,0.61,0.49]
# dwt_data = [0.77,0.33,0.23]
# spncnn_data = [0.89,0.47,0.35]
# dude_data = [0.89,0.49,0.38]
# fusion_rep = [0.94,0.67,0.52]
# dwt_rep = [0.85,0.37,0.26]

# 标签
labels = ['x=0,y=0','x=10,y=10', 'x=20,y=20', 'x=30,y=30', 'x=40,y=40']
# labels = ['no crop','crop 20%', 'crop 30%']

# 创建图形
plt.figure(figsize=(10, 5), frameon=False)  # 设置 frameon=False 来去除边框
# 绘制准确率和AUC折线图
plt.plot(labels, fusion_data, marker='o', label=f'Proposed_PCE')
plt.plot(labels, dwt_data, marker='o', label=f'DWT_PCE')
plt.plot(labels, spncnn_data, marker='o', label=f'SPNCNN_PCE')
plt.plot(labels, dude_data, marker='o', label=f'DFPRNU-NET_PCE')
plt.plot(labels, fusion_rep, marker='s', label=f'Proposed_rep')
plt.plot(labels, dwt_rep, marker='s', label=f'DWT_rep')
# 添加标题和标签
plt.title('Comparison of Accuracy of different method on shift')
plt.xlabel('shift pixel')
plt.ylabel('accuracy')
plt.legend()
# 显示图形
plt.grid(True)
# 保存图形为SVG格式
plt.savefig('comparison_accuracy_shift.svg', format='svg', bbox_inches='tight', pad_inches=0.05,dpi=2000)
plt.show()