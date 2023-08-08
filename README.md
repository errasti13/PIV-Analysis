# Particle Image Velocimetry (PIV) Analysis

Particle Image Velocimetry (PIV) is a powerful technique used in fluid mechanics to visualize and analyze fluid flow patterns. This repository contains a Python script for performing PIV analysis on a sequence of grayscale images, calculating sub-pixel accurate displacements, and generating gradient magnitude values to understand fluid flow behavior.

Three point Gaussian fit and parabolic fit have been implemented for sub pixel accuracy. For data smoothing median blur and Singular Value Decomposition based reconstructions have been implemented.
## Features

- Load and normalize grayscale images for analysis.
- Compute sub-pixel accurate displacements using normalized cross-correlation.
- Perform Singular Value Decomposition (SVD) based smoothing and reconstruction.
- Visualize velocity vector fields, displacement magnitudes, and distribution statistics.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/piv-analysis.git](https://github.com/errasti13/PIV-Analysis.git
   cd PIV-Analysis

2. Prerequisites
   ```bash
   pip install opencv-python numpy matplotlib

3. Run the PIV Script
   ```bash
     python ./Main.py

4. Change the path to analyzed images in function main() in Main.py. More than two images can be analyzed in serial.

## Credits

Author: Jon Errasti Odriozola

Date: September 2021

Last Modified: August 2023

## Contributing
Contributions to this repository are welcome! If you find any issues, or if you have ideas for improvements or additional features, feel free to open a pull request or submit an issue.

## License 
This project is licensed under the MIT License - see the LICENSE file for details


