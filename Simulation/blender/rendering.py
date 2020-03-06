import bpy

# create light datablock, set attributes
light_data = bpy.data.lights.new(name="light_2.80", type='POINT')
light_data.energy = 30

# create new object with our light datablock
light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)

# link light object
bpy.context.collection.objects.link(light_object)

# make it active 
bpy.context.view_layer.objects.active = light_object

#change location
light_object.location = (5, 5, 5)

# update scene, if needed
dg = bpy.context.evaluated_depsgraph_get() 
dg.update()


