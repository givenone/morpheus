import bpy
import os


def render(dir_path, resolution_x, resolution_y) :
    path_dir = bpy.context.scene.render.filepath #save for restore
    scene = bpy.context.scence

    for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
        
        scene.camera = cam
        scene.render.resolution_x = resolution_x
        scene.render.resolution_y = resolution_y
        scene.render.engine = 'CYCLES'
        scene.render.filepath = os.path.join(dir_path, cam.name)
        bpy.ops.render.render(write_still=True)
        
        bpy.context.scene.render.filepath = path_dir