import numpy as np

def calculate_camera_orientation_matrix():
    """
    Calculates and returns the camera orientation matrix based on predefined
    camera direction vectors.

    The function establishes an orthonormal basis for the camera's orientation
    in 3D space, where:
    - Z vector represents the forward direction of the camera.
    - Y vector is initially partially defined, with its complete direction
      determined by ensuring orthogonality with Z.
    - X vector is calculated as the cross product of Y and Z to guarantee
      orthogonality among all vectors.

    Returns:
        camera_matrix (numpy.ndarray): A 3x3 matrix representing the camera's
                                       orientation in 3D space.
    """
    # Define the forward (Z) direction vector of the camera.
    camera_forward = np.array([-0.232, 0.888, 0.398])

    # Initialize the up (Y) direction vector of the camera, missing the z-component.
    camera_up_partial = np.array([0.232, -0.888, 0])

    # Calculate the missing z-component of the up vector to ensure orthogonality with the forward vector.
    up_z_component = -(camera_forward[0] * camera_up_partial[0] + camera_forward[1] * camera_up_partial[1]) / \
                     camera_forward[2]
    camera_up = np.array([camera_up_partial[0], camera_up_partial[1], up_z_component])

    # Normalize the up (Y) and forward (Z) vectors to ensure they are unit vectors.
    normalized_up = camera_up / np.linalg.norm(camera_up)
    normalized_forward = camera_forward / np.linalg.norm(camera_forward)

    # Calculate the right (X) direction vector as the cross product of up (Y) and forward (Z) vectors.
    camera_right = np.cross(normalized_up, normalized_forward)

    # Combine the right (X), up (Y), and forward (Z) vectors into the camera orientation matrix.
    camera_matrix = np.vstack((camera_right, normalized_up, normalized_forward)).T
    return camera_matrix


