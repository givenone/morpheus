# blender --background --python 01_cube.py --render-frame 1 -- </path/to/output/dir> </path/to/input/object>

import preprocessing.dome_obj_data as dome
import blender.blender as bl
import os, sys, util

def get_output_file_path() -> str:
    return str(sys.argv[sys.argv.index('--') + 1])


def get_input_obj_path() -> str:
    return int(sys.argv[sys.argv.index('--') + 2])



if __name__ == "__main__":
    # Args
    output_dir_path = get_output_file_path()
    input_file_path = get_input_obj_path()

    # Setting
    scene = bpy.context.scene

    util.clean_objects()

    vertices = [x[1:] for x in dome.get_dome_vertices()]
    cameras = [{'location': [1,0,0]}]
    bl.displaceLight(vertices)
    bl.displaceObject(input_file_path)
    bl.displaceCamera()

    blender.rendering.render(output_dir_path, 1080, 720)
    
    #camera_object = bpy.data.objects["Camera"]
    #utils.set_output_properties(scene, resolution_percentage, output_file_path)
    #utils.set_cycles_renderer(scene, camera_object, num_samples)

