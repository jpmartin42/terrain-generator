import os
import argparse
import numpy as np
import trimesh
from typing import Optional
import random
from dataclasses import dataclass

from typing import Tuple, Optional

from terrain_generator.wfc.wfc import WFCSolver

from terrain_generator.trimesh_tiles.mesh_parts.create_tiles import create_mesh_pattern, get_mesh_gen

# from trimesh_tiles.mesh_parts.overhanging_parts import FloorOverhangingParts
from terrain_generator.utils.mesh_utils import visualize_mesh, compute_sdf
from terrain_generator.utils import calc_spawnable_locations_with_sdf
from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import (
    MeshPattern,
    OverhangingMeshPartsCfg,
    OverhangingBoxesPartsCfg,
)
from terrain_generator.trimesh_tiles.mesh_parts.overhanging_parts import get_cfg_gen

from configs.navigation_cfg import IndoorNavigationPatternLevels
from configs.overhanging_cfg import OverhangingTerrainPattern, OverhangingPattern, OverhangingFloorPattern
from terrain_generator.navigation.mesh_terrain import MeshTerrain, MeshTerrainCfg
from alive_progress import alive_bar


# Terrain types to include
# - Clutter (misc)
# - Stairs
# - Slope
# - Uneven step

from terrain_generator.trimesh_tiles.mesh_parts.mesh_parts_cfg import (
    OverhangingMeshPartsCfg,
    OverhangingBoxesPartsCfg,
    MeshPattern,
    MeshPartsCfg,
    WallPartsCfg,
    StairMeshPartsCfg,
    PlatformMeshPartsCfg,
    HeightMapMeshPartsCfg,
    FloatingBoxesPartsCfg,
)

from terrain_generator.trimesh_tiles.patterns.pattern_generator import (
    generate_random_box_platform,
    # generate_walls,
    generate_floating_boxes,
    generate_narrow,
    generate_platforms,
    generate_ramp_parts,
    generate_stair_parts,
    generate_stepping_stones,
    generate_floating_capsules,
    generate_random_boxes,
    generate_overhanging_platforms,
    add_capsules,
)

from terrain_generator.trimesh_tiles.mesh_parts.rough_parts import generate_perlin_tile_configs


from terrain_generator.trimesh_tiles.mesh_parts.mountain import generate_perlin_terrain

from examples.generate_with_wfc import solve_with_wfc, create_mesh_from_cfg

@dataclass
class CustomTerrainPattern(MeshPattern):
    dim: Tuple[float, float, float] = (2.0, 2.0, 2.0)  # x, y, z
    seed: int = 1234

    enable_wall: bool = False

    # random box platform
    random_cfgs = []
    n_random_boxes: int = 10
    random_box_weight: float = 0.01
    perlin_weight: float = 0.1

    # random box platform
    random_cfgs = []
    n_random_boxes: int = 10
    random_box_weight: float = 0.01
    perlin_weight: float = 0.1
    for i in range(n_random_boxes):
        random_cfgs += generate_random_box_platform(
            name=f"box_platform_flat_{i}",
            offset=0.0,
            height_diff=0.0,
            height_std=0.1,
            n=6,
            dim=dim,
            weight=random_box_weight / n_random_boxes,
        )
        random_cfgs += generate_perlin_tile_configs(f"perlin_{i}", [2, 2, 2], weight=perlin_weight / n_random_boxes)
        random_cfgs += generate_perlin_tile_configs(
            f"perlin_0.5_{i}", [2, 2, 2], weight=perlin_weight / n_random_boxes, offset=0.5, height=0.5
        )
        random_cfgs += generate_perlin_tile_configs(
            f"perlin_1.0_{i}", [2, 2, 2], weight=perlin_weight / n_random_boxes, offset=0.0, height=1.0
        )
        random_cfgs += generate_perlin_tile_configs(
            f"perlin_1.0_1.0{i}", [2, 2, 2], weight=perlin_weight / n_random_boxes, offset=1.0, height=5.0
        )
    mesh_parts: Tuple[MeshPartsCfg, ...] = (
        (WallPartsCfg(name=f"floor", dim=dim, wall_edges=(), weight=0.01),)
        + tuple(
            generate_platforms(name="platform_1", dim=dim, max_h=1.0, min_h=0.0, weight=0.5, enable_wall=enable_wall)
        )
        + tuple(
            generate_platforms(name="platform_2", dim=dim, max_h=2.0, min_h=0.0, weight=0.5, enable_wall=enable_wall)
        )
        + tuple(
            generate_platforms(name="platform_2_1", dim=dim, max_h=2.0, min_h=1.0, weight=0.5, enable_wall=enable_wall)
        )
        # + tuple(
        #     generate_platforms(name="platform_0.5", dim=dim, max_h=0.5, min_h=0.0, weight=0.5, enable_wall=enable_wall)
        # )
        # + tuple(
        #     generate_platforms(
        #         name="platform_1_0.5", dim=dim, max_h=1.0, min_h=0.5, weight=0.5, enable_wall=enable_wall
        #     )
        # )
        + tuple(random_cfgs)
        + tuple(
            generate_stair_parts(
                name="stair", dim=dim, seed=seed, array_shape=[15, 15], weight=0.5, depth_num=2, enable_wall=enable_wall
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_offset",
                dim=dim,
                seed=seed,
                array_shape=[15, 15],
                weight=2.0,
                depth_num=2,
                offset=1.0,
                enable_wall=enable_wall,
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_low",
                dim=dim,
                total_height=0.5,
                seed=seed,
                array_shape=[15, 15],
                weight=0.5,
                depth_num=2,
                enable_wall=enable_wall,
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_low_offset",
                dim=dim,
                total_height=0.5,
                offset=0.5,
                seed=seed,
                array_shape=[15, 15],
                weight=0.5,
                depth_num=2,
                enable_wall=enable_wall,
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_low_offset_1",
                dim=dim,
                total_height=0.5,
                offset=1.0,
                seed=seed,
                array_shape=[15, 15],
                weight=0.5,
                depth_num=2,
                enable_wall=enable_wall,
            )
        )
        + tuple(
            generate_stair_parts(
                name="stair_low_offset_2",
                dim=dim,
                total_height=0.5,
                offset=1.5,
                seed=seed,
                array_shape=[15, 15],
                weight=0.5,
                depth_num=2,
                enable_wall=enable_wall,
            )
        )
        + tuple(
            generate_ramp_parts(
                name="ramp",
                dim=dim,
                seed=seed,
                array_shape=[30, 30],
                total_height=1.0,
                offset=0.00,
                weight=0.05,
                depth_num=1,
            )
        )
    )


if __name__ == "__main__":

    # Load arguments
    parser = argparse.ArgumentParser(description="Create mesh from configuration")
    parser.add_argument(
        "--cfg",
        type=str,
        choices=["indoor", "overhanging", "overhanging_floor"],
        default="indoor",
        help="Which configuration to use",
    )

    parser.add_argument("--over_cfg", action="store_true", help="Whether to use overhanging configuration")
    parser.add_argument("--visualize", action="store_true", help="Whether to visualize the generated mesh")
    parser.add_argument("--enable_history", action="store_true", help="Whether to enable mesh history")
    parser.add_argument("--enable_sdf", action="store_true", help="Whether to enable sdf")
    parser.add_argument("--initial_tile_name", type=str, default="floor", help="Whether to enable sdf")
    
    parser.add_argument(
        "--mesh_dir", type=str, default="results/terrain_tests", help="Directory to save the generated mesh files")
    parser.add_argument("--mesh_name", type=str, default="mesh", help="Base name of the generated mesh files")

    parser.add_argument("--curriculum", action="store_true", help="Whether to use a curriculum")
    args = parser.parse_args()

    cfg = CustomTerrainPattern()
    over_cfg = None

    for i in range(1):
        mesh_prefix = f"{args.mesh_name}_{i}"
        create_mesh_from_cfg(
            cfg,
            over_cfg,
            prefix=mesh_prefix,
            initial_tile_name=args.initial_tile_name,
            mesh_dir=args.mesh_dir,
            visualize=args.visualize,
            enable_history=args.enable_history,
            enable_sdf=args.enable_sdf,
        )

    
