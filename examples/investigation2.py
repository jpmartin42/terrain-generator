#
# Copyright (c) 2023, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import os
import trimesh
import numpy as np
from typing import Tuple

from terrain_generator.trimesh_tiles.mesh_parts.mountain import generate_perlin_terrain

def generate_mountain(mesh_dir):

    terrain = generate_perlin_terrain(horizontal_scale=0.2, vertical_scale=3.0)
    terrain2 = generate_perlin_terrain(horizontal_scale=0.2, vertical_scale=4.0)


    mesh = terrain
    # mesh.show()
    bbox = mesh.bounding_box.bounds
    # Get the center of the bounding box.
    center = np.mean(bbox, axis=0)
    center[2] = 0.0
    # Translate the mesh to the center of the bounding box.

    mesh = mesh.apply_translation(-center)

    mesh = mesh + terrain2

    os.makedirs(mesh_dir, exist_ok=True)
    mesh.export(os.path.join(mesh_dir, "mountain.obj"))
    terrain.export(os.path.join(mesh_dir, "terrain.obj"))


if __name__ == "__main__":
    mesh_dir = "results/investigation2"
    generate_mountain(mesh_dir)
