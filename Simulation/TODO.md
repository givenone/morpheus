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

- making New Image for scaled face. & format also png & hdr.

- other frames (devebec, gosh, ... )
- camera & light & frame object implementation
- camera & light parameter setting
- object option (shading, texture, ... )
- specular/diffuse(subsurface scattering) separation
- animation (not necessary because we just obtain rendered pictures, but for synchronization purposes using animation would be good.)
- various rendering engine

- light source visibility
- emulating polarizers on simulation
- Geometry alignment using normals (how to mix specular, diffuse normals?)
