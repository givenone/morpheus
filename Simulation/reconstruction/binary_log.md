# log

=======================

- uused smc.freeimage for hdr image processing(https://pypi.org/project/smc.freeimage/) -> installation fail.

- just used imread and mixed normal done (initially, blue normal)

- opencv imread reads color as BGR not RGB.

- How to modulate by Fresnel Gain? : We dont know grazing angle yet.

- [suv space](http://vision.ucsd.edu/~spmallick/research/suv/index.html) : suv 에서 s component (white : specular로 가정) 제외한 uv의 rgb color 를 diffuse color로 가정하고 normal을 계산. 

- We observe that surface details are
clearly visible in the acquired mixed photometric normals, partic-
ularly in the shorter green and blue wavelengths. This implies that
the mixed normals encode a mixture of a diffuse normal and some
specular reflectance, with the contribution of the specular informa-
tion in the mixture being strongest in the blue channel for skin

- Finally, the specular normal ~ N spec can be computed as
the half-vector given the viewing direction ~ V the estimated reflec-
tion vector ~ R spec ????

- specular normal directly obtained with the above
procedure also suffers from significant low-frequency bias (binary로 discrete하게 하기 때문.)
->applying a high-pass filter to ~ N spec and
adding the high-pass to the low-frequency diffuse normal ~ N u,v ob-
tained using color-space separation, and finally re-normalizing the
resultant normal map.

- rotation matrix for suv separation : [link](https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d) [github](https://gist.github.com/asanakoy/c82420afdab4d5eaa78c9f9481e462e1#file-rgb_2_suv-py-L53)

- TODO :: Spectral Multiplexing (not considered yet)

- TODO :: Optimization of the code

- using tensordot, einsum for rotation of all image efficiently