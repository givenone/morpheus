def BinaryPattern(lightList):
    # ('v', x, y, z) -> multiple list of (intensity, x, y, z)
    # Binary Gradient Pattern
    # Diffuse-Specular Separation using Binary Spherical Gradient Illumination
    # Eurographics Symposium on Rendering (EGSR: EI&I) 2018
    # Christos Kampouris, Stefanos Zafeiriou, Abhijeet Ghosh
    # Imperial College London
    x = [(100, x, y, z) if x >= 0 else (0, x, y, z) for (a, x, y, z) in lightList]
    x_c = [(100, x, y, z) if x < 0 else (0, x, y, z) for (a, x, y, z) in lightList]
    y = [(100, x, y, z) if y >= 0 else (0, x, y, z) for (a, x, y, z) in lightList]
    y_c = [(100, x, y, z) if y < 0 else (0, x, y, z) for (a, x, y, z) in lightList]
    z = [(100, x, y, z) if z >= 0 else (0, x, y, z) for (a, x, y, z) in lightList]
    z_c = [(100, x, y, z) if z < 0 else (0, x, y, z) for (a, x, y, z) in lightList]
    w = [(100, x, y, z) for (a, x, y, z) in lightList] # all on (white)
    b = [(0, x, y, z) for (a, x, y, z) in lightList] # all off
    return [('x', x), ('x_c', x_c), ('y', y), ('y_c', y_c), ('z', z), ('z_c', z_c), ('w', w), ('b', b)]

def GradientPattern(lightList) :
    # ('v', x, y, z) -> multiple list of (intensity, x, y, z)
    # Spherical Gradient Pattern by USC Institute for Creative Technologies 
    # Multiview Face Capture using Polarized Spherical Gradient Illumination
    # ACM Trans. on Graphics (Proc. SIGGRAPH Asia) 2011 - Abhijeet Ghosh 

    x = [(-100 * (x-1) / 2, x, y, z) for (a, x, y, z) in lightList]
    #x_c = [(-100 * (x-1 / 2), x, y, z) for (a, x, y, z) in lightList]
    y = [(-100 * (y-1) / 2, x, y, z) for (a, x, y, z) in lightList]
    #y_c = [(-100 * (y-1 / 2), x, y, z) for (a, x, y, z) in lightList]
    z = [(100 * (z+1) / 2, x, y, z) for (a, x, y, z) in lightList]
    #z_c = [(-100 * (z-1 / 2), x, y, z) for (a, x, y, z) in lightList]

    w = [(100, x, y, z) for (a, x, y, z) in lightList]
    #b = [(0, x, y, z) for (a, x, y, z) in lightList]
    return [('x', x), ('y', y), ('z', z),  ('w', w)]
