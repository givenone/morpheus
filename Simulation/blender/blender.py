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
from math import radians, atan, sqrt, acos
import os

def displaceLight(lightList):
    # lightList : [intensity, x, y, z]

    # set background light strength (ambient light)
    bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[1].default_value = 2

    for i, light in enumerate(lightList) :
        # make light data
        light_data = bpy.data.lights.new(name="light_{}".format(i), type='SPOT')
        light_data.spot_size = radians(45)
        light_data.spot_blend = 0.7
        light_data.specular_factor = 0.5 #0.5 ?
        light_data.energy = light[0]
        light_data.use_shadow = True
        
        # make light object
        light_object = bpy.data.objects.new(name="light_{}".format(i), object_data = light_data)
        # link the scene
        bpy.context.collection.objects.link(light_object)

        #bpy.context.view_layer.objects.active = light_objects
        scale = 5
        light_object.location = [scale * x for x in light[1:]]
        getRotation(light_object)
        # TODO :: setup more light properties

    #dg = bpy.context.evaluated_depsgraph_get() 
    #dg.update()

def displaceCamera(cameraList) :
    #cameraList : [location : [x,y,z]]
    for i, camera in enumerate(cameraList) :
        bpy.ops.object.camera_add(location=camera['location'])
        curr = bpy.context.object
        getRotation(curr)
        curr.name = "camera"+str(i)

def displaceObject(file_path) :
    # displace object at (0,0,0)
    # file : obj path. 
    imported_object = bpy.ops.import_scene.obj(filepath=file_path)
    file_name = os.path.basename(file_path).split('.')[0]
    
    obj = bpy.data.objects[file_name] #object name is descripted in .obj file "o" if not, same as file name.
    
    #TODO :: align at center
    (x,y,z) = obj.dimensions
    scale = sqrt(pow(x,2) + pow(y,2) + pow(z,2))
    scale = max(x, y, z)
    obj.dimensions = (x/scale * 2 , y/scale * 2, z/scale * 2) # scale dimensions (size of an object)
    #TODO :: shading, other properties for rendering
    return True

def displaceFrame(vertices, edges, scale) :
    vert = []
    for v in vertices :
        vert.append([x * scale for x in v[1:]])
    ed = []
    for e in edges : 
        e = e[1:]
        for i in range(len(e)) :
            ed.append((e[i]-1, e[(i+1) % len(e)]-1))
    me = bpy.data.meshes.new("Frame")
    me.from_pydata(vert, ed, [])
    me.validate()
    me.update()

    
    ob = bpy.data.objects.new("Frame", me)

    # apply skin modifier
    #bpy.ops.ojbect.select_name(name="Frame")
    bpy.context.collection.objects.link(ob)
    
    bpy.context.view_layer.objects.active = ob

    bpy.ops.object.modifier_add(type='SKIN')
    bpy.ops.object.editmode_toggle()


    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.transform.skin_resize(
        value=(0.2, 0.2, 0.2),
    )
    bpy.ops.object.editmode_toggle()

    # brdf for color

    name = "forframe"
    bpy.data.materials.new(name)
    bpy.data.materials[name].use_nodes = True
    material = bpy.data.materials[name]
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes.new( type = 'ShaderNodeOutputMaterial' )
    diffuse = nodes.new( type = 'ShaderNodeBsdfDiffuse' )
    
    link = links.new( diffuse.outputs['BSDF'], output.inputs['Surface'] )

    x = nodes.get('Principled BSDF')
    x.inputs['Base Color'].default_value = [0.19, 0.19, 0.19, 1]
    x.inputs['Specular'].default_value = 0.3
    x.inputs['Roughness'].default_value = 0.3
    x.inputs['Metallic'].default_value = 0.3
    bpy.data.objects["Frame"].active_material = bpy.data.materials[name]
    
 
def displaceRoom(scale) :
    return

def getRotation(blender_object) :
    (x, y, z) = blender_object.location
    blender_object.rotation_mode = "AXIS_ANGLE"
    w = acos(z/sqrt(pow(x,2)+pow(y,2)+pow(z,2)))
    blender_object.rotation_axis_angle = [w, -y, x, 0]
 