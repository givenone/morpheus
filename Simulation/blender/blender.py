# blender python module
import bpy

import os, sys

from math import radians, atan, sqrt, acos
import configparser
import ast

conf_path = os.path.abspath(os.path.dirname(__file__))
conf_path = os.path.join(conf_path, "simulation.conf")
configuration = configparser.ConfigParser()
configuration.read(conf_path)
configuration = configuration['simulation']

def displaceLight(lightList, frame_scale):
    # lightList : [intensity, x, y, z]

    # set background light strength (ambient light)
    bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[1].default_value = float(configuration['background'])

    for i, light in enumerate(lightList) :
        # make light data
        light_data = bpy.data.lights.new(name="light_{}".format(i), type='SPOT')
        light_data.spot_size = radians(int(configuration['spot_size']))
        light_data.spot_blend = float(configuration['blend'])
        light_data.specular_factor = float(configuration['specular_factor'])
        light_data.energy = light[0]
        light_data.use_shadow = True
        light_data.use_custom_distance = True
        light_data.cutoff_distance = int(configuration['cutoff_distance'])
        
        # make light object
        light_object = bpy.data.objects.new(name="light_{}".format(i), object_data = light_data)
        # link the scene
        bpy.context.collection.objects.link(light_object)

        #bpy.context.view_layer.objects.active = light_objects
        scale = frame_scale
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
        curr.data.sensor_width = int(configuration['sensor_width'])
        getRotation(curr)
        curr.name = "camera"+str(i)

def displaceObject(file_path) :
    # displace object at (0,0,0)
    # file : obj path. 
    imported_object = bpy.ops.import_scene.obj(filepath=file_path)
    file_name = os.path.basename(file_path).split('.')[0]
    
    obj = bpy.data.objects[file_name] #object name is descripted in .obj file "o" if not, same as file name.
    
    #TODO :: align at center
    obj.location = ast.literal_eval(configuration['obj_location'])

    (x,y,z) = obj.dimensions
    scale = sqrt(pow(x,2) + pow(y,2) + pow(z,2))
    scale = max(x, y, z)
    obj.dimensions = (x/scale * 2 , y/scale * 2, z/scale * 2) # scale dimensions (size of an object)
    #TODO :: shading, other properties for rendering
    

def displaceBlenderObject(file_path, obj_name) :
    bpy.ops.wm.append(filename = obj_name, directory = file_path)
    obj = bpy.data.objects[obj_name]

    #TODO :: align at center
    obj.location = ast.literal_eval(configuration['obj_location'])
    (x,y,z) = obj.dimensions
    scale = sqrt(pow(x,2) + pow(y,2) + pow(z,2))
    scale = max(x, y, z)
    obj.dimensions = (x/scale * 2 , y/scale * 2, z/scale * 2) # scale dimensions (size of an object)



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

    #output = nodes.new( type = 'ShaderNodeOutputMaterial' )
    #diffuse = nodes.new( type = 'ShaderNodeBsdfDiffuse' )
    
    #link = links.new( diffuse.outputs['BSDF'], output.inputs['Surface'] )

    x = nodes.get('Principled BSDF')
    x.inputs['Base Color'].default_value = ast.literal_eval(configuration['f_Base_Color'])
    x.inputs['Specular'].default_value = float(configuration['f_Specular'])
    x.inputs['Roughness'].default_value = float(configuration['f_Roughness'])
    x.inputs['Metallic'].default_value = float(configuration['f_Metallic'])
    bpy.data.objects["Frame"].active_material = bpy.data.materials[name]
    
 
def displaceRoom(scale) :
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.selected_objects[0]

    cube.dimensions = (scale, scale, scale)
    name = "room"
    bpy.data.materials.new(name)
    bpy.data.materials[name].use_nodes = True
    material = bpy.data.materials[name]
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    x = nodes.get('Principled BSDF')
    x.inputs['Base Color'].default_value = ast.literal_eval(configuration['r_Base_Color'])
    x.inputs['Specular'].default_value = float(configuration['r_Specular'])
    x.inputs['Roughness'].default_value = float(configuration['r_Roughness'])
    x.inputs['Metallic'].default_value = float(configuration['r_Metallic'])
    x.inputs['Sheen'].default_value = float(configuration['r_Sheen'])
    cube.active_material = bpy.data.materials[name]

def getRotation(blender_object) :
    (x, y, z) = blender_object.location
    blender_object.rotation_mode = "AXIS_ANGLE"
    w = acos(z/sqrt(pow(x,2)+pow(y,2)+pow(z,2)))
    blender_object.rotation_axis_angle = [w, -y, x, 0]
 