import numpy as np

arr = np.array([[[[1], [2]], [[3], [4]]]])

print(arr)

new = np.pad(arr, ((0, 0), (1, 1), (1, 1), (0, 0)), 'reflect')

print(new[0, :, :, 0])
