import numpy as np

dt = np.fromfunction(lambda i, j:i + j, (4, 5), dtype=int)
dt2 = np.fromfunction(lambda i, j:i + j, (6, 8), dtype=int)
print(dt)
print(dt2)
array2 = np.pad(dt,((0,dt2.shape[0]-dt.shape[0]),(0,dt2.shape[1]-dt.shape[1])),'constant')
print(array2)