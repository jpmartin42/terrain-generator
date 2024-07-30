#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import inspect

from terrain_generator.wfc.wfc import WFCSolver

from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import create_mesh_pattern
from terrain_generator.utils import visualize_mesh
from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import MeshPartsCfg, MeshPattern

from terrain_generator.trimesh_tiles.mesh_parts.rough_parts import generate_perlin_tile_configs


from configs.indoor_cfg import IndoorPattern, IndoorPatternLevels
from configs.navigation_cfg import IndoorNavigationPatternLevels
from alive_progress import alive_bar

from typing import Tuple
from terrain_generator.trimesh_tiles.primitive_course.steps import *
from terrain_generator.utils import convert_heightfield_to_trimesh
import random

from perlin_numpy import generate_perlin_noise_2d, generate_fractal_noise_2d

# Greyscale image generation fun
from PIL import Image

# Function to encode max height INTO image.
# Pixel [0,1] = 0 always as "base" reference
# Pixel [0,0] = max_height >= 5m
# Pixel [1,0] = max_height >= 10m
# Pixel [2,0] = max_height >= 20m
# Pixel [3,0] = max_height >= 30m
# Pixel [4,0] = max_height >= 40m
# Pixel [5,0] = max height == 50m

def encode_max_heights(terrain_height):
    hmax = terrain_height.max()

    max_height = 5

    max_level = 1

    if(hmax >= 50):
        print("Oh no, max height exceeds max value!")
    elif(hmax > 40):
        max_height = 50
        max_level = 6
    elif(hmax > 30):
        max_height = 40
        max_level = 5

    elif(hmax > 20):
        max_height = 30
        max_level = 4

    elif(hmax > 10):
        max_height = 20
        max_level = 3

    elif(hmax > 5):
        max_height = 10
        max_level = 2

    for i in range(max_level):
        terrain_height[i,0] = np.array(max_height).astype(np.uint8)

    for i in range(5, max_level-1, -1):
        terrain_height[i,0] = np.array(0).astype(np.uint8)


    terrain_height[0,1] = np.array(0).astype(np.uint8)

    return terrain_height, max_height

def generate_perlin2(
    mesh_dir, 
    base_shape: Tuple = (256, 256),
    base_res: Tuple = (16, 16),
    base_octaves: int = 2,
    base_fractal_weight: float = 0.2,
    noise_res: Tuple = (4, 4),
    noise_octaves: int = 5,
    base_scale: float = 2.0,
    noise_scale: float = 1.0,
    horizontal_scale: float = 1.0,
    vertical_scale: float = 10.0,
    height_diff: float = 0.0, # slope
    folder_name = "perlin_flat",
    level: float = 0.0,
    max_height = 10,
    file_index = 0
):
    # Generate fractal noise instead of Perlin noise
    base = generate_perlin_noise_2d(base_shape, base_res, tileable=(True, True))
    base += generate_fractal_noise_2d(base_shape, noise_res, base_octaves, tileable=(True, True)) * base_fractal_weight

    slope_list = []

    # Create a height array for offsets
    slope_scale = height_diff / float(base_shape[1])
    slope_vec = None

    if height_diff != 0:
        slope_vec = np.arange(0.0, height_diff, slope_scale)
    else:
        slope_vec = np.zeros(base_shape[1])

    for _ in range(base_shape[0]):
        slope_list.append(slope_vec)
    
    slope_array = np.asarray(slope_list)

    base += slope_array

    # Use different weights for the base and noise heightfields
    noise = generate_fractal_noise_2d(base_shape, noise_res, noise_octaves, tileable=(True, True))
    
    terrain_height = base * base_scale + noise * noise_scale

    terrain_height, hmax = encode_max_heights(terrain_height)

    # print(base.max())
    # print(noise.max(),"\n")

    terrain_array = (((terrain_height - terrain_height.min()) / (terrain_height.max() - terrain_height.min())) * 255.9).astype(np.uint8)

    # Set standard values for corner. It'll look weird but it guarantees that the height is the desired value
    # terrain_array[0,0] = np.array(255.9).astype(np.uint8)
    # terrain_array[0,1] = np.array(0).astype(np.uint8)

    img_name = f"mesh_{level:.2f}_{file_index:03}.png"

    img = Image.fromarray(terrain_array)
    img.save(mesh_dir+"/"+folder_name+"/"+img_name)

def generate_box_grid(
    mesh_dir, 
    n = 8,    
    base_shape: Tuple = (256, 256),
    height_diff: float = 0.0, # slope
    height_std: float = 0.0,
    folder_name = "box_grid",
    level: float = 0.0,
    max_height = 10,
    file_index = 0
):
    terrain_height = np.zeros(base_shape)

    box_size = int(base_shape[1]/n)

    slope = height_diff/n

    for i in range(n):
        for j in range(n):

            # Generate a random std for this deviation
            randval = random.uniform(0, 1)

            # print("(", j*box_size,":",(j+1)*box_size - 1,")   ,   (" , (i)*box_size , ":", (i+1)*box_size - 1 ,   ")     -     ", i * slope + randval*height_std)

            terrain_height[(j*box_size) : ((j+1)*box_size ), (i*box_size) : ((i+1)*box_size )] = i * slope + randval*height_std

            # print(i * slope + randval*height_std)

    terrain_height, hmax = encode_max_heights(terrain_height)

    terrain_array = (((terrain_height - terrain_height.min()) / (terrain_height.max() - terrain_height.min())) * 255.9).astype(np.uint8)

    # Set standard values for corner. It'll look weird but it guarantees that the height is the desired value
    # terrain_array[0,0] = np.array(255.9).astype(np.uint8)
    # terrain_array[0,1] = np.array(0).astype(np.uint8)

    img_name = f"mesh_{level:.2f}_{file_index:03}.png"

    img = Image.fromarray(terrain_array)
    img.save(mesh_dir+"/"+folder_name+"/"+img_name)
    

if __name__ == "__main__":

    np.random.seed(0)
    
    mesh_dir = "results/training_terrains"

    # Train with max height of at most 25 degrees
    length     = 50
    height_max = 50
    

    for level in np.arange(0.0, 1.05, 0.05):
    # for level in [1.0]:
        for i in range(10):

            scale = random.randint(4, 8)
            perlin_res = int(256 / int(2**(scale)))

            # print(perlin_res)

            hmax = random.uniform(0.5, 1) * level * height_max

            # Generate flat perlin terrain
            generate_perlin2(mesh_dir=mesh_dir, 
                            base_shape=(256,256),
                            base_res = (perlin_res,perlin_res),
                            height_diff=0.0,
                            vertical_scale=1,
                            horizontal_scale=0.0234375,
                            folder_name="00",
                            level = level,
                            base_scale = level*(2**(scale-4)),
                            noise_scale = level*(2**(scale-4)),
                            max_height = height_max,
                            file_index=i)
            
            scale = random.randint(0, 4)
            perlin_res = int(256 / int(2**(scale)))

            hmax = random.uniform(0.5, 1) * level * height_max
            res = random.randint(1, 3)

            # Sloped perlin terrains
            generate_perlin2(mesh_dir=mesh_dir, 
                            base_shape=(256,256),
                            base_res=(perlin_res,perlin_res),
                            height_diff=hmax,
                            vertical_scale=1,
                            horizontal_scale=0.0234375,
                            folder_name="01",
                            level = level,
                            base_scale = level*(2**(scale-4)),
                            noise_scale = level*(2**(scale-4)),
                            max_height = height_max,
                            file_index=i)
            
            hmax = random.uniform(0.5, 1) * level * height_max
            res = random.randint(6, 8)

            # Sloped box grids
            generate_box_grid(mesh_dir=mesh_dir,
                            n = 2**res,    
                            base_shape = (4096,4096),
                            height_diff = hmax, # slope
                            height_std = 0.3*level,
                            folder_name = "02",
                            level = level,
                            max_height = height_max,
                            file_index=i)

            hmax = random.uniform(0.5, 1) * level * height_max
            res = random.randint(6, 8)

            # Normal box grids
            generate_box_grid(mesh_dir=mesh_dir,
                            n = 2**res,    
                            base_shape = (4096,4096),
                            height_diff = 0, # slope
                            height_std = 0.5*level,
                            folder_name = "03",
                            level = level,
                            max_height = height_max,
                            file_index=i)
            
            hmax = random.uniform(0.5, 1) * level * height_max
            res = random.randint(6, 8)

            # Stairs
            generate_box_grid(mesh_dir=mesh_dir,
                            n = 2**res,    
                            base_shape = (4096,4096),
                            height_diff = hmax, # slope
                            height_std = 0.0,
                            folder_name = "04",
                            level = level,
                            max_height = height_max,
                            file_index=i)

            hmax =  level * height_max

            # Constant slope
            generate_box_grid(mesh_dir=mesh_dir,
                            n = 256,                  # For a constant slope, need grid <= height resolution
                            base_shape = (256,256),
                            height_diff = hmax, # slope
                            height_std = 0,
                            folder_name = "05",
                            level = level,
                            max_height = height_max,
                            file_index=i)
