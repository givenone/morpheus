import bpy
import os


def render(dir_path) :
    path_dir = bpy.context.scene.render.filepath #save for restore
    scene = bpy.context.scene

    for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
        
        scene.camera = cam
        scene.render.engine = 'CYCLES' # rendering engine
        scene.render.filepath = os.path.join(dir_path, cam.name)
        bpy.ops.render.render(write_still=True)
        
        bpy.context.scene.render.filepath = path_dir