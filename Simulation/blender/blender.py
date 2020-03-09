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
from math import radians, atan
import os


def displaceLight(lightList):
    # lightList : [x, y, z, scale, intensity]

    # set background light strength (ambient light)
    bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[1].default_value = 2

    for i, light in enumerate(lightList) :
        # make light data
        light_data = bpy.data.lights.new(name="light_{}".format(i), type='SPOT')
        light_data.spot_size = radians(80)
        light_data.spot_blend = 0.15
        light_data.specular_factor = 1 #0.5 ?
        light_data.energy = 100
        light_data.use_shadow = True
        
        # make light object
        light_object = bpy.data.objects.new(name="light_{}".format(i), object_data = light_data)

        # link the scene
        bpy.context.collection.objects.link(light_object)

        #bpy.context.view_layer.objects.active = light_object
        scale = 5
        light_object.location = [scale * x for x in light[:3]]
        light_object.rotation_euler = getRotation(light[:3])
        # TODO :: setup more light properties

    #dg = bpy.context.evaluated_depsgraph_get() 
    #dg.update()

def displaceCamera(cameraList) :
    return True

def displaceObject(file_path) :
    # displace object at (0,0,0)
    # file : obj path. 
    imported_object = bpy.ops.import_scene.obj(filepath=file_path)
    file_name = os.path.basename(file_path).split('.')[0]
    bpy.data.objects[file_name].dimensions = (1, 1, 1) # scale dimensions (size of an object)
    
    return True

def getRotation(location) :
    if location[2] == 0 :
        return (0, radians(90), atan(location[1]/location[0]))
    return (-atan(location[0]/location[2]), atan(location[1]/location[2]), 0)


