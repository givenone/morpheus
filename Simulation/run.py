# blender --background --python 01_cube.py --render-frame 1 -- </path/to/output/dir> </path/to/input/object>
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
    output_dir_path = get_output_file_path()
    input_file_path = get_input_obj_path()

    # Setting
    scene = bpy.context.scene

    util.clean_objects()

    #TODO
    vertices = preprocessing.get_dome_vertices()
    lights = blender.BinaryPattern(vertices) if option == "BINARY" else blender.GradientPattern(vertices)
    cameras = [{'location': [0,-3,0]}]

    for light in lights :
        pattern_name = light[0]
        blender.displaceLight(light[1])
        blender.displaceObject(input_file_path)
        blender.displaceCamera(cameras)

        blender.rendering.render(output_dir_path + "/" + pattern_name)
    
    #camera_object = bpy.data.objects["Camera"]
    #utils.set_output_properties(scene, resolution_percentage, output_file_path)
    #utils.set_cycles_renderer(scene, camera_object, num_samples)

