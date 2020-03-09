import bpy

def clean_objects() -> None:
    for item in bpy.data.objects:
        bpy.data.objects.remove(item)