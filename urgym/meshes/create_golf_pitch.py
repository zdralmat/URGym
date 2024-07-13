import trimesh
import numpy as np

# Create a cube
cube = trimesh.creation.box(extents=[0.8, 0.8, 0.05])

# Create a cylinder to subtract from the cube (the hole)
cylinder = trimesh.creation.cylinder(radius=0.02, height=0.2)

# Position the cylinder in the center of the cube
cylinder.apply_translation([0, 0, 0])

# Subtract the cylinder from the cube to create the hole
cube_with_hole = cube.difference(cylinder)

# Export the result to an OBJ file
cube_with_hole.export('golf_pitch.obj')

print("Golf pitch created successfully!")
