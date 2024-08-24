import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the size of the voxel grid
grid_size = 5

# Create an empty grid
voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)


# Define a small 3x3x3 cube in the center of the grid
voxel_grid[1:4, 1:4, 1:4] = True
print(voxel_grid)
# Plotting the voxel grid
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the voxel plot
ax.voxels(voxel_grid, edgecolor='k')

# Set the labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Display the plot
plt.show()
