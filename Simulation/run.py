import preprocessing.dome_obj_data as dome
import blender.blender as bl

vertices = [x[1:] for x in dome.get_dome_vertices()]
bl.displaceLight(vertices)
