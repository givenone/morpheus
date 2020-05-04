import bpy
import os


def render(dir_path, filename) :
    path_dir = bpy.context.scene.render.filepath #save for restore
    scene = bpy.context.scene

    # resolution : 4K
    scene.render.resolution_x = 2160#4096
    scene.render.resolution_y = 3840#2160

    # HDRI Rendering

    bpy.context.scene.render.image_settings.file_format = "PNG"
    
    for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
        
        scene.camera = cam
        #scene.render.engine = 'CYCLES' # rendering engine
        scene.render.filepath = os.path.join(dir_path, filename)
        bpy.ops.render.render(write_still=True)
        
        bpy.context.scene.render.filepath = path_dir