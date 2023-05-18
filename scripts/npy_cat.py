import sys
import numpy as np

numpy_file = sys.argv[1]
numpy_list = np.load(numpy_file, allow_pickle=True)

for item in numpy_list:
    print(item)
