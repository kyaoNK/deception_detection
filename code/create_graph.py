import sys
import os
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = "../train_result/"

argv = sys.argv

filename = argv[0]
path = os.path.join(RESULT_DIR, filename)
data_set = np.loadtxt(fname=path, dtype="float", delimiter=",")

y_select = argv[1]

plt.xlabel("epoch")
x = data_set[:, 0]

if y_select == "loss":
    plt.ylabel("loss")
    y = data_set[:,1]
else if y_select == "train":
    plt.ylabel("train acc")
    y = data_set[:,2]
else if y_select == "wolf":
    plt.ylabel("wolf acc")
    y = data_set[:,3]
else y_select == "human":
    plt.ylabel("human acc")
    y = data_set[:,4]

plt.plot(x, y)
plt.show()