from collections import namedtuple

# Define a namedtuple for SphereData
SphereDataTuple = namedtuple("SphereDataTuple", ["name", "position", "color"])

class SphereData:
    """
    Class to represent data for a sphere.

    Attributes:
    - name (str): The name of the sphere.
    - position (tuple): The position of the sphere as a tuple (x, y, z).
    - colour (tuple): The color of the sphere as an RGB tuple (r, g, b).
    """

    def __init__(self, name, position, color):
        """
        Initialize a SphereData object.

        Args:
        - name (str): The name of the sphere.
        - position (tuple): The position of the sphere as a tuple (x, y, z).
        - colour (tuple): The color of the sphere as an RGB tuple (r, g, b).
        """
        self.name = name
        self.position = position
        self.color = color

    def __str__(self):
        """
        Return a string representation of the SphereData object.

        Returns:
        - str: A string representation of the SphereData object.
        """
        return f"{self.name}: Position {self.position}, Color {self.color}"
