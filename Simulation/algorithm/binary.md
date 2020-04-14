
Development Log
======

- python with 구문 : [link](https://cjh5414.github.io/python-with/)

- .hdr file :  32-bit floating point numbers, for each channel.

- [HSV](https://darkpgmr.tistory.com/66) [HSV_detailed](https://en.wikipedia.org/wiki/HSL_and_HSV)

EGSR (Rapid Acquisition of Specular and Diffuse Normal Maps from Polarized Spherical Gradient Illumination Wan-Chun Ma)

- to infer BRDFs across the entire object surface

- spatially-varying BRDFs

- Thus, our four lighting patterns, each photographed under two polarization states, yield the diffuse normal (per color channel), diffuse albedo (per color channel), specular nor-mal, and specular intensity for each pixel.

- Since our technique estimates just the normals and albedo, we use a manually-chosen specular roughness parameter for our renderings

- differences between the specular
and diffuse normals are important for characterizing the re-
flectance of organic materials. From this observation, we
present a real-time hybrid normal shading technique that
uses independent normal maps for the specular channel and
each of the three diffuse color channels.

- combined with cost-effective
structured light scanning to produce 3D models

- centroid of brdf ? (BRDFs are
largely symmetrical around the normal or reflected direction
for diffuse and specular BRDFs respectively.)

- How to compute Fresnel Gain?

- do not wish to limit the light place-
ment to a horizontal plane, we must create a spherical direc-
tion field of linear polarization for the lights designed so that
the light reflected specularly in accordance with Equation (Fresnel's Equation) toward the camera viewpoint will be vertically polarized re-gardless of the angle of incidence (i.e. which light it origi-
nated from)

- *** vertical 에 따라, polarizer angle needs to be varied.

- the intensity of the reflected spec-
ular component from a surface element oriented along the
halfway vector depends not only on the specular albedo ρ s
but also on the incident light direction itself as specified by
the Fresnel equations. This effect must be divided out when
interpreting the reflected specular light as ρ s . Near the Brew-
ster angle, the specular reflection becomes very weak, mak-
ing recovery of ρ s unreliable for these angles.

- A disadvantage of separating reflectance components with
linear polarization is that the required polarization direction
field is highly dependent on the viewing direction 
-> Circular polarizers have chirality but not direction. Distribut-
ing circular polarizers of the same chirality over a sphere
provides a polarization field which is not only smoothly
varying but also symmetric with respect to choice of camera
viewpoint. This preserves the viewpoint independence estab-
lished by using linear gradient patterns in the first place.

- Instead, we take advantage of the fact that r s and r p are
roughly equal for moderate angles of incidence, causing the
polarization of the specularly reflected light to remain nearly
circular. Capturing only two images, one with a circular po-
larizer and another with the same filter placed backwards,
then allows specular and diffuse components to be separated
in the same way as for the linearly polarized case

- Normal Map Acquisition for Highly Specular
Surfaces :: The discrete lighting sphere used so far cannot estimate sur-
face normals for very shiny surfaces since most surface ori-
entations will reflect little or no light from the sources. To
apply our technique to a highly specular surface, we use
a continuous-field hemispherical lighting device

- 결국 어떤 Polarizer 를 어떻게 Setting할 것이고 (Maybe Circular Polarizer?), angle에 따른 Fresnel Gain 을 계산하는 방법을 더 공부해야할 필요가 있음.

Diffuse-Specular Separation using Binary Spherical Gradient Illumination (Christos Kampouris, Stefanos Zafeiriou, Abhijeet Ghosh / Imperial College London)

- diffuse-specular separation of albedo and photometric normals without
requiring polarization using binary spherical gradient illumination

- Mallick et al. [MZKB05] proposed
a color-space separation technique for dielectric materials that
linearly transforms RGB color into a formulated suv space where
s aligns with white (color of light) and u, v align with chroma.
(They employ the u, v components for improved photometric stereo
discarding specularities)

- The LED sphere also has an addi-
tional set of 168 RGB LED lamps (ColorKinetics iColor MR gen3)
mounted adjacent to the white LED lamps (RGB lamps for spectral multiplexing)

- a surface on a
convex dielectric object exhibits pure diffuse reflectance when lit
with the dark (off) hemisphere while exhibiting both diffuse and
specular reflectance when lit with the bright (on) hemisphere of
the binary gradients.

- for colored di-
electric materials, specular reflectance is white in color when a sur-
face is lit with white light. Hence, when a colored dielectric sur-
face is lit with a binary spherical gradient and its complement, the
reflected RGB color of the surface is more saturated in the darker
(pure diffuse) lighting condition while exhibiting reduced satura-
tion under the brighter (diffuse + specular) lighting condition due
to mixing with some white reflection.

- Diffuse Normal : uv space normal -> consistency between dielectic normals => diffuse normal

- The
normalized halfway vector between  r and  v corresponds to the specular normal direction

- diffuse normal 을 diffuse albedo based modulation ?

- high pass filter to specular normal