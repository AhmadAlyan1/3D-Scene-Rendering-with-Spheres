# 3D Scene Rendering with Spheres

This project demonstrates rendering a 3D scene with spheres onto a 2D image. It includes functions for:

- Creating a blank image.
- Creating and rendering spheres.
- Calculating lighting effects.
- Sorting spheres by proximity to the camera.
- Applying gamma correction to the final image.

## 📌 Features

- **3D Sphere Rendering**: Simulates a simple 3D environment projected onto a 2D plane.
- **Lighting Effects**: Basic shading and light intensity calculations.
- **Camera System**: Computes orientation matrix for proper perspective rendering.
- **Optimized Sorting**: Ensures proper sphere rendering order.

## 📂 Project Structure

```
📦 3D-Scene-Rendering
├── 📜 camera.py      # Handles camera orientation and transformations
├── 📜 sphere.py      # Defines the SphereData class with properties
├── 📜 main.py        # Main script to create and render the 3D scene
├── 📜 README.md      # Project documentation
```

## 🚀 Installation & Usage

### 1️⃣ Install Dependencies
Make sure you have Python installed, then run:
Install:
```sh
pip install pillow numpy
```

### 2️⃣ Run the Project
Execute the main script to generate the 3D scene:
```sh
python main.py
```

## 🔧 Requirements
- Python 3.x
- NumPy
- Pillow (PIL)
---
