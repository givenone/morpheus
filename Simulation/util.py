import bpy

def setting(**kwargs) :
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES' # rendering engine
    return

def clean_objects() -> None:
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)