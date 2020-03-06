# blender python module
import bpy

"""
def displaceGeomtry(pathToOBJ, pathForExport):
    scene = bpy.context.screen.scene
    for object_ in scene.objects:
        bpy.data.objects.remove(object_, True)

    imported_object = bpy.ops.import_scene.obj(filepath=pathToOBJ)
    obj_object = bpy.context.selected_objects[0]
    bpy.context.scene.objects.active = obj_object

    for item in bpy.data.materials:
        #Enable "use_shadeless"
        item.use_shadeless = True

    subd = obj_object.modifiers.new("subd", type='SUBSURF')
    # # subd.levels = 2
    bpy.ops.object.modifier_apply(modifier=subd.name)

    tex = obj_object.active_material.active_texture
    dispMod = obj_object.modifiers.new("Displace", type='DISPLACE')
    dispMod.texture = tex
    dispMod.texture_coords = "UV"
    dispMod.strength = -0.002
    bpy.ops.object.modifier_apply(modifier=dispMod.name)

    bpy.ops.export_scene.obj(filepath=pathForExport)

if __name__ == "__main__":
    displaceGeomtry("forBlender.obj",
    "fromBlender.obj")
"""
from math import radians


def displaceLight(lightList):
    for i, light in enumerate(lightList) :
        light_data = bpy.data.lights.new(name="light_{}".format(i), type='SpotLight')
        light_data.spot_size = radians(80)
        light_data.blend = 0.15
        light_data.specular = 1 #0.5 ?
        light_object = bpy.data.objects.new(name="light_{}".format(i), object_data = light_data)
        bpy.context.view_layer.objects.active = light_object
        light_object.location = lightList[:3]
        light_object.angle
        # TODO :: orientation, angle
    dg = bpy.context.evaluated_depsgraph_get() 
    dg.update()
