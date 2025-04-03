"""
Students names and ids:
    Ahmad Alyan, 212159933
    Lubaba Neiroukh,

This project demonstrates rendering a 3D scene with spheres onto a 2D image. It includes functions for
 creating a blank image, creating and rendering spheres, calculating lighting, sorting spheres by proximity
  to the camera, and applying gamma correction to the final image.
"""

from camera import calculate_camera_orientation_matrix
from PIL import Image, ImageDraw
from Sphere import SphereData
import numpy as np
import math

# Constants defining the scene and camera
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

# Number of the visible faces points of the sphere
POINTS_NUMBER = [
    [0, 1, 2, 3, 4]  # five points
]

# Define points for a unit sphere
SPHERE_POINTS = np.array([[-0.5, 0, 0],  # index 0 (left of sphere)
                          [0, 0.5, 0],  # index 1 (top of sphere)
                          [0.5, 0, 0],  # index 2 (right of sphere)
                          [0, -0.5, 0],  # index 3 (bottom of sphere)
                          [0, 0, 0]  # index 4 (center of sphere)
                          ])

# Projection matrix for 3D to 2D projection
PROJECTION_MATRIX = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])


def create_blank_image(height, width):
    """
    Create a blank image with a specified height and width.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        numpy.ndarray: A NumPy array representing the blank image.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[0:height, 0:width] = BACKGROUND_COLOR
    return image


def create_sphere(center, image, color):
    """
    Create a 3D sphere and render it onto the image.

    Args:
        center (numpy.ndarray): The center position of the sphere in 3D space.
        image (PIL.Image.Image): The PIL image onto which the sphere will be rendered.
        color (numpy.ndarray): The RGB color of the sphere.

    Returns:
        None
    """
    # Calculate the 3D coordinates of the sphere's points based on the center
    transformed_points_3d = SPHERE_POINTS + center.reshape(1, -1)

    # Project the 3D points to 2D screen coordinates
    projected_points = np.dot(PROJECTION_MATRIX, np.dot(CAMERA_ORIENTATION_MATRIX, transformed_points_3d.T)).T[:, :2]

    # Scale and position the projected points on the image
    projected_points = (projected_points * [X_SCALE, -Y_SCALE]) + IMAGE_CENTER - 50

    # Render the faces of the sphere onto the image
    render_sphere_faces(transformed_points_3d, center, projected_points, color, image)


def render_sphere_faces(transformed_points, sphere_center, projected_points, color, image):
    """
    Render the faces of the sphere onto the image.

    Args:
        transformed_points (numpy.ndarray): The transformed 3D points of the sphere.
        sphere_center (numpy.ndarray): The center position of the sphere in 3D space.
        projected_points (numpy.ndarray): The projected 2D points of the sphere.
        color (numpy.ndarray): The RGB color of the sphere.
        image (PIL.Image.Image): The PIL image onto which the faces will be rendered.

    Returns:
        None
    """
    # Iterate over the face indices to render each face of the sphere
    for face_indices in POINTS_NUMBER:
        # Get the 2D points of the current face
        face_points_2d = [projected_points[i] for i in face_indices]
        # Draw the visible face onto the image
        draw_visible_face(image, face_points_2d, color)


def draw_visible_face(image, face_points_2d, color):
    """
    Draw a visible face of the sphere onto the image.

    Args:
        image (PIL.Image.Image): The PIL image onto which the face will be drawn.
        face_points_2d (numpy.ndarray): The 2D points representing the face on the image.
        color (numpy.ndarray): The RGB color of the sphere.

    Returns:
        None
    """
    draw = ImageDraw.Draw(image)
    face_points = np.array(face_points_2d, dtype=int)

    # Define the shading ellipse for the sphere's shadow
    shade = [(face_points[1][0] - 250, face_points[1][1]),
             (face_points[1][0] + 170, face_points[1][1] + 170)]
    draw.ellipse(shade, fill=(int(0.07 * 255), int(0.07 * 255), int(0.07 * 255)))

    # Get the pixel position of the sphere's center
    sphere_pixel_pos = face_points_2d[4]

    # Create a meshgrid around the sphere's center to iterate over pixels
    xx, yy = np.meshgrid(np.arange(int(sphere_pixel_pos[0] - SPHERE_RADIUS), int(sphere_pixel_pos[0] + SPHERE_RADIUS)),
                          np.arange(int(sphere_pixel_pos[1] - SPHERE_RADIUS), int(sphere_pixel_pos[1] + SPHERE_RADIUS)))

    # Calculate distance of each pixel from the sphere's center
    distance = (xx - sphere_pixel_pos[0]) ** 2 + (yy - sphere_pixel_pos[1]) ** 2

    # Mask pixels based on distance to only iterate over those within the sphere's radius
    mask = distance <= SPHERE_RADIUS ** 2
    xx_masked = xx[mask]
    yy_masked = yy[mask]

    # Iterate over masked pixels and calculate lighting and shading
    for x, y in zip(xx_masked, yy_masked):
        normal = calculate_normal(x, y, sphere_pixel_pos)
        illumination = calculate_lighting(normal, LIGHT_SOURCE)
        shaded_color = tuple(int(illumination * c) for c in color)
        draw.point((x, y), fill=shaded_color)


def calculate_normal(x, y, sphere_pixel_pos):
    """
    Calculate the normal vector at a given pixel position on the sphere's surface.

    Args:
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.
        sphere_pixel_pos (numpy.ndarray): The 2D position of the center of the sphere's face.

    Returns:
        numpy.ndarray: The calculated normal vector.
    """
    # Calculate the components of the normal vector
    nx = (x - sphere_pixel_pos[0]) / SPHERE_RADIUS
    ny = (y - sphere_pixel_pos[1]) / SPHERE_RADIUS
    nz = math.sqrt(SPHERE_RADIUS ** 2 - (x - sphere_pixel_pos[0]) ** 2 - (y - sphere_pixel_pos[1]) ** 2) / SPHERE_RADIUS
    # Normalize the normal vector
    normal = np.array([nx, ny, nz]) / np.linalg.norm(np.array([nx, ny, nz]))
    return normal


def calculate_lighting(normal, light_position, light_intensity=1):
    """
    Calculate the lighting intensity at a point on the sphere's surface.

    Args:
        normal (numpy.ndarray): The normal vector at the surface point.
        light_position (numpy.ndarray): The position of the light source in 3D space.
        light_intensity (float, optional): The intensity of the light source. Defaults to 1.

    Returns:
        float: The calculated illumination intensity.
    """
    # Calculate the direction from the surface point to the light source
    light_direction = np.array(light_position) - np.array([CAMERA_POSITION[0], CAMERA_POSITION[1], CAMERA_POSITION[2]])

    # Calculate the distance between the surface point and the light source
    light_distance = np.linalg.norm(light_direction)

    # Normalize the light direction vector
    light_direction /= light_distance

    # Calculate the illumination intensity based on the dot product of the normal and light direction vectors
    illumination = np.dot(normal, light_direction)

    # Clamp the illumination to a range between 0.05 and 1 and scale it by the light intensity
    illumination = max(0.05, min(1, illumination)) * light_intensity

    return illumination


def calculate_distance(sphere, camera_position):
    """
    Calculate the Euclidean distance between a sphere and the camera.

    Args:
        sphere (SphereData): The sphere object containing its position.
        camera_position (numpy.ndarray): The position of the camera in 3D space.

    Returns:
        float: The calculated distance between the sphere and the camera.
    """
    # Convert sphere and camera positions to NumPy arrays for calculations
    sphere_pos = np.array([sphere.position[0], sphere.position[1], sphere.position[2]])
    camera_pos = np.array([camera_position[0], camera_position[1], camera_position[2]])
    # Calculate the Euclidean distance between the sphere and the camera
    distance = np.linalg.norm(sphere_pos - camera_pos)
    return distance


def sort_spheres_by_proximity_to_camera(camera_position, sphere_list):
    """
    Sort a list of spheres based on their proximity to the camera.

    Args:
        camera_position (numpy.ndarray): The position of the camera in 3D space.
        sphere_list (list): A list of SphereData objects representing spheres in the scene.

    Returns:
        list: The sorted list of SphereData objects.
    """
    # Calculate distances from each sphere to the camera
    distances = np.array([calculate_distance(sphere, camera_position) for sphere in sphere_list])
    # Get the indices that would sort the distances array
    sorted_indices = np.argsort(distances)
    # Rearrange sphere_list based on the sorted indices
    sorted_spheres = [sphere_list[i] for i in sorted_indices]
    return sorted_spheres


def apply_gamma_correction(image_array):
    """
    Apply gamma correction to an image array.

    Args:
        image_array (numpy.ndarray): The image array to be corrected.

    Returns:
        numpy.ndarray: The gamma-corrected image array.
    """
    # Normalize the image to the range [0, 1]
    normalized_image = image_array / 255.0

    # Apply gamma correction by taking the square root
    gamma_corrected_image = np.sqrt(normalized_image)

    # Rescale the gamma-corrected image back to the range [0, 255]
    rescaled_image = gamma_corrected_image * 255.0

    # Clip the image values to ensure they are within the valid range [0, 255]
    rescaled_image = np.clip(rescaled_image, 0, 255).astype(np.uint8)

    return rescaled_image


if __name__ == "__main__":
    # Create a blank image
    image = create_blank_image(IMAGE_HEIGHT, IMAGE_WIDTH)

    # Define spheres
    SPHERE1 = SphereData(name="SPHERE1", position=np.array((-0.86, 2.18, 0.5)),
                         color=np.array([0.3, 0.7, 0.2]) * 255)  # GREEN
    SPHERE2 = SphereData(name="SPHERE2", position=np.array((0.54, 1.8, 0.5)),
                         color=np.array([0.4, 0.4, 0.6]) * 255)  # BLUE
    SPHERE3 = SphereData(name="SPHERE3", position=np.array((0.27, 0.41, 0.5)),
                         color=np.array([0.8, 0.7, 0.4]) * 255)  # ORANGE
    SPHERE4 = SphereData(name="SPHERE4", position=np.array((-0.99, 0.7, 0.5)),
                         color=np.array([0.8, 0.225, 0.046]) * 255)  # RED
    SPHERES = [SPHERE1, SPHERE2, SPHERE3, SPHERE4]

    # Sort the spheres based on their distance from the camera
    order_to_draw = sort_spheres_by_proximity_to_camera(CAMERA_POSITION, SPHERES)
    image = Image.fromarray(image, 'RGB')

    # Render each sphere onto the image
    for sphere in order_to_draw:
        create_sphere(sphere.position, image, sphere.color)

    # Apply gamma correction to the image
    image_array_corrected = np.array(image)
    corrected_image_array = apply_gamma_correction(image_array_corrected)
    corrected_image = Image.fromarray(corrected_image_array, 'RGB')

    corrected_image.save("4_Spheres.png")
    corrected_image.show()