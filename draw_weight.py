# import matplotlib.pyplot as plt
# import numpy as np
#
# # 假设你的权重矩阵是一个NumPy数组
# weight_matrix = np.random.rand(10, 13)  # 替换为你的实际权重矩阵
#
# plt.imshow(weight_matrix, cmap='viridis', interpolation='nearest')
# plt.colorbar()
# plt.title('Weight Matrix Visualization')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 假设你的权重矩阵是一个NumPy数组
weight_matrix = np.random.rand(15, 10)  # 替换为你的实际权重矩阵

plt.imshow(weight_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar().remove()  # 移除颜色条
plt.title('Weight Matrix Visualization')

# 保存图像，不包含颜色条
plt.savefig('weight_matrix_visualization.png', bbox_inches='tight', pad_inches=0.1)
plt.show()