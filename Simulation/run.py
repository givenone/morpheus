#       der-frame 1 -- </path/to/output/dir> </path/to/input/object>
# "/home/givenone/morpheus/photogeometric/Simulation/emily.blend/Object"
import bpy
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import util
import preprocessing
import blender
import geometry

option = "BINARY"

def get_output_file_path() -> str:
    return str(sys.argv[sys.argv.index('--') + 1])


def get_input_obj_path() -> str:
    return str(sys.argv[sys.argv.index('--') + 2])


if __name__ == "__main__":
    # Args
    #output_dir_path = get_output_file_path()
    #input_file_path = get_input_obj_path()

    # Setting
    scene = bpy.context.scene

    util.clean_objects(all = True)
    #util.setting()
    
    frame = "models/dome/ico_3.obj"
    vertices = preprocessing.read_vertices_objects(frame)
    faces = preprocessing.read_faces_objects(frame)

    blender.displaceFrame(vertices,faces, 5) # 5 : scale
    blender.displaceRoom(10) # 10 : scale
    lights = blender.BinaryPattern(vertices) if option == "BINARY" else blender.GradientPattern(vertices)
    
    cameras = [{'location': [0,-4.9,0]}]
    blender.displaceCamera(cameras)
    
    # Displace Object
    blender.displaceBlenderObject("/home/givenone/morpheus/photogeometric/Simulation/emily.blend/Object", "Emily_2_1")

    print("Done preprocessing")

    for light in lights :        
        util.clean_objects() # Remove all lights
        pattern_name = light[0] # name of pattern.
        blender.displaceLight(light[1])

        print("Light Displacement Done")
        
        blender.rendering.render("/home/givenone/morpheus/photogeometric/Simulation/output", pattern_name)
        print(pattern_name, "rendering done")
             
    print("Rendering Done")

    util.save_configuration("config.txt")  # Saving Configuration Details   

    # Genearte Geometry Details
    (w, h) = geometry.depth.getDistanceMap(1, 1, "/home/givenone/morpheus/photogeometric/Simulation/output", "dist")
    geometry.pointcloud.generate_pointcloud()
    print("Generated Point Cloud")

    # Reconstruction   