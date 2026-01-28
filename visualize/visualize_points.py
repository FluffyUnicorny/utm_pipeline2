import numpy as np
import matplotlib.pyplot as plt

POINTS_PATH = "manual_output/points3d_clean_sfm.npy"

pts = np.load(POINTS_PATH)

print("Points shape:", pts.shape)

x = pts[:,0]
y = pts[:,1]
z = pts[:,2]

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')

ax.scatter(x, y, z, s=1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
