# get depth map of object.
import bpy
import numpy as np
import os, sys
#import cv2

def getDistanceMap(width, height, dir_path, filename):

    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links
    # input render layer node
    rl = tree.nodes['Render Layers']
    
    # output node
    v = tree.nodes['Composite']
    v.use_alpha = False

    # Links
    links.new(rl.outputs['Depth'], v.inputs[0]) # link Z to output

    scene.render.resolution_x = width  #2160
    scene.render.resolution_y = height #3840
    bpy.context.scene.render.image_settings.file_format = "HDR"

    w, h = (36, 24)
    # render
    
    for i, cam in enumerate([obj for obj in bpy.data.objects if obj.type == 'CAMERA']):

        w, h  = cam.data.sensor_width, cam.data.sensor_height

        scene.camera = cam
        scene.render.filepath = os.path.join(dir_path, filename + str(i))
        bpy.ops.render.render(write_still=True)
    
    scene.use_nodes = False

    return(w * 0.001, h * 0.001) # mm scale

    #pixels = bpy.data.images['Viewer Node'].pixels
    #depth = np.asarray(pixels)
    #depth = np.reshape(depth, (height, width, 4))
    #depth = depth[:, :, 0]
    #np.savetxt(os.path.join(dir_path, filename) + ".txt", depth)
    #cv2.imwrite(os.path.join(dir_path, filename) + ".exr", depth)