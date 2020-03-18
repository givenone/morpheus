import bpy

def setting(**kwargs) :
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES' # rendering engine
    bpy.context.scene.cycles.device = "GPU" # GPU support
    bpy.context.scene.cycles.progressive = "BRANCHED_PATH"
    return

def clean_objects(ty="LIGHT", all=False) -> None:
    for item in bpy.data.objects:
        if all or item.type == ty :
            bpy.data.objects.remove(item)