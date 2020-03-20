#       der-frame 1 -- </path/to/output/dir> </path/to/input/object>
# "/home/givenone/morpheus/photogeometric/Simulation/emily.blend/Object"
import bpy
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import util
import preprocessing
import blender

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
    #TODO
    frame = "models/dome/ico_3.obj"
    vertices = preprocessing.read_vertices_objects(frame)
    faces = preprocessing.read_faces_objects(frame)
    blender.displaceFrame(vertices,faces, 5)
    blender.displaceRoom(20)
    lights = blender.BinaryPattern(vertices) if option == "BINARY" else blender.GradientPattern(vertices)
    print("Done preprocessing")
    cameras = [{'location': [0,-3,0]}]

    for light in lights :
        util.clean_objects() # Remove all lights
        pattern_name = light[0]

        blender.displaceLight(light[1])
        #blender.displaceObject(input_file_path)
        print("Light Displacement Done")
        blender.displaceBlenderObject("/home/givenone/morpheus/photogeometric/Simulation/emily.blend/Object", "Emily_2_1")
        blender.displaceCamera(cameras)
        break
        #blender.rendering.render(output_dir_path + "/" + pattern_name)
           