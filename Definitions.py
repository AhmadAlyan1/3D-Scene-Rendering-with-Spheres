"""
3D Scene Configuration for Rendering

This script sets up essential parameters for rendering a 3D scene using a virtual camera and a light source. It defines the camera's position, the light source's location, and various other settings crucial for generating the final image.

Variables:
- CAMERA_POSITION: NumPy array defining the XYZ coordinates of the camera in the scene.
- LIGHT_SOURCE: NumPy array indicating the XYZ coordinates of the light source.
- BACKGROUND_COLOR: RGB color of the background, scaled to 255 (greyscale by default).
- IMAGE_HEIGHT, IMAGE_WIDTH: Dimensions of the output image in pixels.
- IMAGE_CENTER: XY coordinates of the image's center, used for aligning the scene.
- X_SCALE, Y_SCALE: Scaling factors for adjusting the scene's geometry in the X and Y directions.
- SPHERE_RADIUS: Radius of the sphere to be rendered in the scene.
- CAMERA_ORIENTATION_MATRIX: A matrix calculated to determine the camera's orientation based on its position and target.

These settings are utilized to compute the visual representation of objects within the scene, taking into account the camera's perspective, the influence of the light source, and geometric transformations.
"""
from camera import calculate_camera_orientation_matrix
import numpy as np


CAMERA_POSITION = np.array((-0.232, 0.888, 0.398))
LIGHT_SOURCE = np.array((-0.218, 0.073, 0.973))
BACKGROUND_COLOR = np.array([0.5, 0.5, 0.5]) * 255
IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
IMAGE_CENTER = [IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2]
X_SCALE = 350
Y_SCALE = 600
SPHERE_RADIUS = 200
CAMERA_ORIENTATION_MATRIX = calculate_camera_orientation_matrix()
