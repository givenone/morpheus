It consists of 3 parts.

1. preprocessing
   * Take coordinates and intensities of lights , and lighting patterns.
   * It uses properties & models 
2. blender
   * using bpy library, render objects under lights.
   * Get photos from designated viewpoints
3. reconstruction
   * make normals, albedos. Also merge with 3d geometry to make a complete 3D viewable file.


- Need to implement command line program (using package management,... to automatically run all parts)
- Testing for various lighting condition