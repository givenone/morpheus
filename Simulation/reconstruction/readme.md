# Reconstruction

Get input as rendered image from simulation, reconstruct albedo and normal maps. Specular & Diffuse separation is also executed.

- binary_reconstruction.py : reconstruction based on binary gradient patterns. It is now written on python, but needs to be re-factored using c++ & opencv for optimization. This comprises following parts

1. Read geometry
2. Mixed Albedo
3. Specualr albedo using hsv color space (Fresnel Coefficient is not modulated now).
4. Diffuse Albedo
5. Mixed normal
6. Diffuse normal (using suv space)
7. Specular normal (with high-pass filter)

**You need to specify input images directory in __main__ @ binary_reconstruction.py and file format before run**