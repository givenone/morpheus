# blender --background --python 01_cube.py --render-frame 1 -- </path/to/output/dir> </path/to/input/object>
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
    output_dir_path = get_output_file_path()
    form = get_input_obj_path()

    # Setting
    scene = bpy.context.scene

    util.clean_objects(all = True)
    util.setting()
    #TODO
    frame = "models\\dome\\ico_3.obj"
    vertices = preprocessing.read_vertices_objects(frame)
    faces = preprocessing.read_faces_objects(frame)
    blender.displaceFrame(vertices,faces, 5)
    blender.displaceRoom(10)
    lights = blender.BinaryPattern(vertices) if option == "BINARY" else blender.GradientPattern(vertices)
    
    cameras = [{'location': [0,-4.9,0]}]
    blender.displaceCamera(cameras)
    
    blender.displaceBlenderObject("C:\\Users\\user\\Desktop\\lightstage\\morpheus\\Simulation\\emily.blend\\Object", "Emily_2_1")

    print("Done preprocessing")
    
    for light in lights :        
        util.clean_objects() # Remove all lights
        pattern_name = light[0]
        #if pattern_name is not 'b' : 
        #    continue
        blender.displaceLight(light[1])
        #blender.displaceObject(input_file_path)
        print("Light Displacement Done")
        
        blender.rendering.render(output_dir_path, pattern_name, form)

        print(pattern_name, "rendering done")      
        #blender.rende1q2wert67y0p-[]74,.