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

def get_output_file_path():
    return str(sys.argv[sys.argv.index('--') + 1])


def get_input_obj_path():
    return str(sys.argv[sys.argv.index('--') + 2])


if __name__ == "__main__":
    # Args
    #output_dir_path = get_output_file_path()
    #input_file_path = get_input_obj_path()

    # Setting
    scene = bpy.context.scene

    util.clean_objects(all = True)
    #util.setting()
    
    # Configuration
    frame = "models/dome/ico_3.obj"
    emily = "/home/givenone/morpheus/photogeometric/Simulation/emily.blend/Object"
    emily_name = "Emily_2_1"
    frame_scale = 5
    room_scale = 10
    camera_location = [0,-4.9,0]
    output_path = "/home/givenone/morpheus/photogeometric/Simulation/output"

    vertices = preprocessing.read_vertices_objects(frame)
    faces = preprocessing.read_faces_objects(frame)

    blender.displaceFrame(vertices,faces, frame_scale)
    blender.displaceRoom(room_scale) # 10 : scale
    lights = blender.BinaryPattern(vertices) if option == "BINARY" else blender.GradientPattern(vertices)
    
    cameras = [{'location': camera_location}]
    blender.displaceCamera(cameras)
    
    # Displace Object
    blender.displaceBlenderObject(emily, emily_name)

    print("Done preprocessing")

    for light in lights :        
        util.clean_objects() # Remove all lights
        pattern_name = light[0] # name of pattern.
        blender.displaceLight(light[1])

        print("Light Displacement Done")
        
        #blender.rendering.render(output_path, pattern_name)
        print(pattern_name, "rendering done")
             
    print("Rendering Done")

    util.save_configuration("config.txt")  # Saving Configuration Details   

    # Genearte Geometry Details
    (w, h) = geometry.depth.getDistanceMap(2160, 3840, output_path, "dist")
    print(w,h)
    #geometry.pointcloud.generate_pointcloud()
    #print("Generated Point Cloud")

    # Reconstruction   