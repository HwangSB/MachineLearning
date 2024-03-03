import os
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("TkAgg")

plt.figure(0)
for dirpath, dirnames, filenames in os.walk("out/train/"):
    for i, filename in enumerate(sorted(filenames), 1):
        with open(os.path.join(dirpath, filename), "r") as f:
            lines = f.readlines()
            x = [list(map(float, line.strip().split()[:2])) for line in lines]
            y = [int(line.strip().split()[2]) for line in lines]

        plt.subplot(8, 7, i)
        plt.title(filename, fontsize=6)
        plt.plot([x[0] for i, x in enumerate(x) if y[i] == 1], [x[1] for i, x in enumerate(x) if y[i] == 1], "bo")
        plt.plot([x[0] for i, x in enumerate(x) if y[i] == 0], [x[1] for i, x in enumerate(x) if y[i] == 0], "ro")

with open("out/test/result", "r") as f:
    lines = f.readlines()
    x = [list(map(float, line.strip().split()[:2])) for line in lines]
    y = [int(line.strip().split()[2]) for line in lines]

plt.figure(1)
plt.plot([x[0] for i, x in enumerate(x) if y[i] == 1], [x[1] for i, x in enumerate(x) if y[i] == 1], "bo")
plt.plot([x[0] for i, x in enumerate(x) if y[i] == 0], [x[1] for i, x in enumerate(x) if y[i] == 0], "ro")
plt.show()
