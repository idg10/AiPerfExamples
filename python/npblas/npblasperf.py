import time
from os import environ

# NOTE: we need to set OMP_NUM_THREADS before loading the numpy module 
# This presumes we're using OpenBLAS, which is what the whl files available
# for numpy are built with.
environ["OMP_NUM_THREADS"] = "16"
import numpy as np

np.show_config()


# size of arrays
n = 2000
# create an array of random values
data1 = np.random.rand(n, n)
data2 = np.random.rand(n, n)

best = 100000
for i in range(20):
    # matrix multiplication
    # t0 = time.time()
    # result = data1.dot(data2)
    # t1 = time.time()
    # print(f"Time (data.dot): {t1 - t0}")

    t0 = time.time()
    result = np.matmul(data1, data2)
    t1 = time.time()
    print(f"Time (np.matmul): {(t1 - t0)*1000}ms")
    best = min(best, (t1 - t0) * 1000)

print()
print(f"Best time: {best}ms")
