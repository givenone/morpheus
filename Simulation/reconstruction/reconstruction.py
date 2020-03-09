import argparse
import time
import xml.etree.ElementTree as ET

import cv2 as cv

import lxml.builder
import numpy as np
import PIL
from lxml import etree
from numpy import array
from PIL import Image
from sklearn.preprocessing import normalize
from toolz.dicttoolz import valmap

import cv2 as cv

# from multiprocessing import Pool

import concurrent.futures

start_time = time.time()

NumberOfCameras = 0

def unitVector(vector):
    return vector / np.linalg.norm(vector)

def angleBetween(v1, v2):
    v1_u = unitVector(v1)
    v2_u = unitVector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def createYRotationMatric(theta):
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])

def calculateMixedNormals():

    for card in range(1, 7):

        images = []

        prefix = "./normalSets6/card" + str(card) + "/"

        names = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg"]

        names = [prefix + name for name in names]

        for i in names:
            img = Image.open(i)
            arr = array(img)
            images.append(arr.astype('float64'))

        height, width, _ = images[0].shape

        N_x = (images[0] - images[1]) / 255
        N_y = (images[2] - images[3]) / 255
        N_z = (images[4] - images[5]) / 255

        encodedImage = np.empty_like(N_x).astype('float64')

        encodedImage[..., 0] = N_x[..., 2]
        encodedImage[..., 1] = N_y[..., 2]
        encodedImage[..., 2] = N_z[..., 2]

        for h in range(height):
            normalize(encodedImage[h], copy=False)

        # only for visualising
        encodedImage = (encodedImage + 1.0) / 2.0
        encodedImage *= 255.0

        im = Image.fromarray(encodedImage.astype('uint8'))
        im.save("encoded{}.jpg".format(card))

def calculateDiffuseNormals(card):

    # for card in range(3, 4):

    images = []

    prefix = "./card{}/".format(card)

    names = [prefix + str(name) + ".TIF" for name in range(3, 16, 2)]
    names.remove(prefix + "11.TIF")

    # print(names)

    for i in names:
        img = Image.open(i)
        arr = array(img)
        images.append(arr.astype('float64'))

    height, width, _ = images[0].shape

    N_x = (images[0] - images[1]) / 255
    N_y = (images[2] - images[3]) / 255
    N_z = (images[4] - images[5]) / 255

    encodedImage = np.empty_like(N_x).astype('float64')

    encodedImage[..., 0] = N_x[..., 0]
    encodedImage[..., 1] = N_y[..., 0]
    encodedImage[..., 2] = N_z[..., 0]

    for h in range(height):
        normalize(encodedImage[h], copy=False)

    # only for visualising
    encodedImage = (encodedImage + 1.0) / 2.0
    encodedImage *= 255.0

    im = Image.fromarray(encodedImage.astype('uint8'))
    im.save("diffuseNormal{}.png".format(card))

def rotationMatrix(path):
    viewVectors = getTranslationVectorPerCamera(path)
    viewVectors = valmap(lambda arr: arr[:3], viewVectors)
    camera3 = viewVectors['card3JPG']
    camera5 = viewVectors['card5JPG']
    angle = angleBetween(np.array(camera3), np.array(camera5))
    rotationMatrix = createYRotationMatric(np.deg2rad(180))

    print(np.rad2deg(-angle))

    return rotationMatrix

def calculateSpecularNormals(card):

    viewVectors = getTranslationVectorPerCamera('blocksExchange.xml')
    viewVectors = valmap(lambda arr: arr[:3], viewVectors)
    camera3 = viewVectors['card3']
    camera5 = viewVectors['card5']
    angle = angleBetween(np.array(camera3), np.array(camera5))
    rotationMatrix = createYRotationMatric(-angle)

    correctedViewVector = np.dot(rotationMatrix, viewVectors['card{}'.format(card)])
    correctedViewVector = correctedViewVector.tolist()
    correctedViewVector[2] *= -1

    correctedViewVector = unitVector(correctedViewVector)

    images = []

    prefix = "./card{}/".format(card)

    xGradients = [prefix + str(name) + ".TIF" for name in range(3, 7)]
    yGradients = [prefix + str(name) + ".TIF" for name in range(7, 11)]
    zGradients = [prefix + str(name) + ".TIF" for name in range(13, 17)]

    names = xGradients + yGradients + zGradients

    for i in names:
        img = Image.open(i)
        arr = array(img)
        images.append(arr.astype('float64'))

    height, width, _ = images[0].shape

    xImages = images[:4]
    yImages = images[4:8]
    zImages = images[8:]

    specularXImages = [xImages[1] - xImages[0], xImages[3] - xImages[2]]
    specularYImages = [yImages[1] - yImages[0], yImages[3] - yImages[2]]
    specularZImages = [zImages[1] - zImages[0], zImages[3] - zImages[2]]

    images = specularXImages + specularYImages + specularZImages
    images = np.array(images)
    images = np.clip(images, 0, 255)

    height, width, _ = images[0].shape

    N_x = (images[0] - images[1]) / 255
    N_y = (images[2] - images[3]) / 255
    N_z = (images[4] - images[5]) / 255

    encodedImage = np.empty_like(N_x).astype('float64')

    encodedImage[..., 0] = N_x[..., 1]
    encodedImage[..., 1] = N_y[..., 1]
    encodedImage[..., 2] = N_z[..., 1]

    for h in range(height):
        normalize(encodedImage[h], copy=False)

    encodedImage[..., 0] = encodedImage[..., 0] + correctedViewVector[0]
    encodedImage[..., 1] = encodedImage[..., 1] + correctedViewVector[1]
    encodedImage[..., 2] = encodedImage[..., 2] + correctedViewVector[2]

    for h in range(height):
        normalize(encodedImage[h], copy=False)

    encodedImage = (encodedImage + 1.0) / 2.0
    encodedImage *= 255.0

    encodedImage = np.clip(encodedImage, 0, 255)

    im = Image.fromarray(encodedImage.astype('uint8'))
    im.save("specularNormal{}.png".format(card), "PNG")

def specularHack():

    for card in range(1, 11):
        diffuse = Image.open("diffuseNormals/diffuseNormal{}.png".format(card))
        specular = Image.open("specularNormals/specularNormal{}.png".format(card))
        blurredSpecular = specular.filter(PIL.ImageFilter.GaussianBlur(radius=10))


        # im = Image.fromarray(blurredSpecular.astype('uint8'))
        # blurredSpecular.save("blurredSpecular{}.png".format(card), "PNG")

        diffuse = np.array(diffuse)
        specular = np.array(specular)
        blurredSpecular = np.array(blurredSpecular)

        highPassFilter = (specular - blurredSpecular)
        highPassFilter = np.clip(highPassFilter, 0, 255) 

        im = Image.fromarray(highPassFilter.astype('uint8'))
        im.save("highPassFilter{}.png".format(card), "PNG")

        newSpec = highPassFilter + diffuse
        newSpec = np.clip(newSpec, 0, 255) 

        height, width, _ = newSpec.shape
        for h in range(height):
            normalize(newSpec[h], copy=False)

        im = Image.fromarray(newSpec.astype('uint8'))
        im.save("newSpec{}.png".format(card), "PNG")


def getPhotoXMLBlock(pathToBlockExchangeXML):
    tree = ET.parse(pathToBlockExchangeXML)
    root = tree.getroot()
    block = root.find('Block')
    photoGroups = block.find('Photogroups').findall('Photogroup')
    photos = map(lambda photogroup: photogroup.findall('Photo'), photoGroups)
    photos = reduce(lambda x,y: x+y, photos)
    return photos

def getCameraName(photoTag):
    imagePath = photoTag.find('ImagePath').text
    name = imagePath.split('/')[-1].split('.')[0]
    return name

def getTranslationVectorPerCamera(pathToBlockExchangeXML):
    photos = getPhotoXMLBlock(pathToBlockExchangeXML)
    NumberOfCameras = len(photos)

    vectorPerCamera = {}

    for photo in photos:
        name = getCameraName(photo)

        center = photo.find('Pose').find('Center')

        #may need to negate coord value for meshlab project
        coords = map(lambda axis: float(axis.text), center)
        vectorPerCamera[name] = coords
        vectorPerCamera[name].append(1)

    return vectorPerCamera

def getRotationMatrixPerCamera(pathToBundlerOut, pathToBlockExchangeXML):
    photos = getPhotoXMLBlock(pathToBlockExchangeXML)
    NumberOfCameras = len(photos)

    rotationMatricPerCamera = {}

    with open(pathToBundlerOut) as f:
        f.readline()
        f.readline()
        f.readline()

        for i in range(1,NumberOfCameras+1):
            card = "card{}".format(i)

            matrix = ""
            for j in range(3):
                matrix += f.readline()
                matrix += "0 "

            f.readline()
            f.readline()
            matrix += "0 0 0 1"

            rotationMatricPerCamera[card] = matrix

    return rotationMatricPerCamera

def getFocalFromAgisoftXml(pathToAgisoftXML):
    tree = ET.parse(pathToAgisoftXML)
    root = tree.getroot()
    cameras = root.find('chunk').find('cameras')

    photoToCamera = {}

    for camera in cameras:
        photoToCamera[camera.get('label')] = camera.get('sensor_id')

    sensors = root.find('chunk').find('sensors')
    sensorFocalLength = {}

    for sensor in sensors:
        sensorId = sensor.get('id')
        sensorFocalLength[sensorId] = sensor.find('calibration').find('f').text

    focalmmPerPhoto = valmap(lambda sensorId: sensorFocalLength[sensorId], photoToCamera)

    return focalmmPerPhoto

def getCameraParameters(pathToBlockExchangeXML, pathToAgisoftXML):
    tree = ET.parse(pathToBlockExchangeXML)
    root = tree.getroot()
    block = root.find('Block')
    photoGroups = block.find('Photogroups').findall('Photogroup')
    counter = 1

    cardToPhotoGroup = {}

    for group in photoGroups:
        photos = group.findall('Photo')
        photos = map(getCameraName, photos)
        for photo in photos:
            cardToPhotoGroup[photo] = "photogoup{}".format(counter)
        counter += 1

    counter = 1

    photogoupToCameraParameters = {}

    for group in photoGroups:
        params = {}
        imageDimensions = map(lambda dim: dim.text, list(group.find('ImageDimensions')))
        distortion = group.find('Distortion')
        distortions = "{} {}".format(distortion.find('K2').text, distortion.find('K3').text)

        params['ViewportPx'] = "{} {}".format(imageDimensions[0], imageDimensions[1])
        params['LensDistortion'] = distortions
        params['CenterPx'] = "{} {}".format(int(imageDimensions[0])/2 , int(imageDimensions[1])/2)
        params['CameraType'] = '0'
        params['PixelSizeMm'] = "1 1"

        photogoupToCameraParameters["photogoup{}".format(counter)] = params

        counter += 1

    focalmmPerPhoto = getFocalFromAgisoftXml(pathToAgisoftXML)

    cardParams = valmap(lambda photogroup: photogoupToCameraParameters[photogroup], cardToPhotoGroup)

    for card in cardParams:
        focalLength = focalmmPerPhoto[card]
        cardParams[card]['FocalMm'] = focalLength

    return cardParams

def createMeshLabXML(name, objectName, ntype):
    E = lxml.builder.ElementMaker()
    print(ntype)

    project = E.MeshLabProject(
        E.MeshGroup(
            E.MLMesh(
                E.MLMatrix44("""
1 0 0 0 
0 1 0 0 
0 0 1 0 
0 0 0 1 
"""),
                label="{}.obj".format(objectName), filename="{}.obj".format(objectName)
            )
        ),
        E.RasterGroup(
            *createVCGTags(ntype)
        )
    )

    tree = etree.ElementTree(project)

    with open("{}.mlp".format(name), "wb") as f:
        f.write("<!DOCTYPE MeshLabDocument>\n")
        tree.write(f, pretty_print=True)

def createVCGTags(ntype):
    translationVectors = getTranslationVectorPerCamera('blocksExchange.xml')
    rotationMatricies = getRotationMatrixPerCamera('bundler.out', 'blocksExchange.xml')
    cameraParams = getCameraParameters('blocksExchange.xml', 'agisoftXML.xml')
    focalLengths = getFocalFromAgisoftXml('agisoftXML.xml')

    cards = sorted(map(lambda x: int(x.replace('card', '')), rotationMatricies.keys()))
    cards = map(lambda x: "card{}".format(x), cards)

    E = lxml.builder.ElementMaker()

    rasterTags = []

    for card in cards:
        number = int(card.replace("card", ""))
        rotationMatrix = rotationMatricies[card].replace('\n', " ")

        translationVector = translationVectors[card]
        translationVector = map(lambda x: x*-1, translationVector)
        translationVector[-1] *= -1
        translationVector = reduce(lambda x, y: "{} {}".format(x,y), translationVector)

        cameraParam = cameraParams[card]
        focalLength = focalLengths[card]
        paramKeys = cameraParam.keys()

        file_name = "{}Normal{}".format(ntype, number)

        tag = E.MLRaster(
             etree.Element('VCGCamera',
             RotationMatrix=rotationMatrix,
             ViewportPx=cameraParam['ViewportPx'],
             CameraType=cameraParam['CameraType'],
             LensDistortion=cameraParam['LensDistortion'],
             PixelSizeMm=cameraParam['PixelSizeMm'],
             CenterPx=cameraParam['CenterPx'],
             FocalMm=cameraParam['FocalMm'],
             TranslationVector=translationVector),

             E.Plane(semantic="1",
             fileName="{}.png".format(file_name)),

             label="{}".format(file_name)
        )

        rasterTags.append(tag)

    return rasterTags


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--maps', action="store_true", default=False)
    parser.add_argument('--diffuseProj', action="store_true", default=False)
    parser.add_argument('--specularProj', action="store_true", default=False)

    args = parser.parse_args()

    if args.maps:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(calculateDiffuseNormals, range(1, 11))

        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(calculateSpecularNormals, range(1, 11))

    if args.diffuseProj:
        createMeshLabXML("diffuseProject", "agisoftExport", "diffuse")

    if args.specularProj:
        createMeshLabXML("specularProject", "diffuseEmbossed", "specular")

    print("--- %s seconds ---" % (time.time() - start_time))