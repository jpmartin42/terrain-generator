import numpy as np

base_shape = (10,10)

terrain_height = np.zeros(base_shape)

print(terrain_height,"\n")

rows = [2]
cols = [0]
terrain_height[2:8,0:5] = 1
terrain_height[8:10,:] = 3

print(terrain_height,"\n")

terrain_array = (((terrain_height - terrain_height.min()) / (terrain_height.max() - terrain_height.min())) * 255.9).astype(np.uint8)

print(terrain_array,"\n")

