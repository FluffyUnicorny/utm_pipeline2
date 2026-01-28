import numpy as np
import matplotlib.pyplot as plt

pts = np.load("data/dataset_umb_h2/sfm/points3d.npy")
cams = np.load("data/dataset_umb_h2/sfm/camera_centers.npy")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Point cloud
ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1)

# Camera trajectory
print("cams shape:", cams.shape)
print(cams)

ax.plot(cams[:,0], cams[:,1], cams[:,2], marker='o')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
