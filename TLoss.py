import numpy as np
import yaml
import os
import zipfile
import shutil

CamMatDict = {}
FullCamDict = {}

os.chdir("Training Images")
ImgCount = len(os.listdir())

for l in range(1, ImgCount + 1):
    # creates var TFileName for name "01", "02"... of each unzipped file
    if l < 10:
        TFileName = "0" + str(l)
    else:
        TFileName = str(l)

    # unzip image file
    with zipfile.ZipFile("t-less_v2_train_kinect_" + TFileName + ".zip", "r") as zip_ref:
        zip_ref.extractall()

    # open gt.yml file and save its data as list
    with open(TFileName + "/gt.yml") as f:
        try:
            ImgData = yaml.load(f, Loader=yaml.BaseLoader)
        except yaml.YAMLError as exc:
            print(exc)

    # iterate through list, access all "cam_R_m2c" values, and organize each into a 3x3 matrix
    for i in range(len(ImgData)):
        for j in ImgData[str(i)]:
            CamMatDict.update({i: np.mat(j["cam_R_m2c"]).reshape(3, 3)})
    FullCamDict.update({l: CamMatDict})
    # delete unzipped files
    shutil.rmtree(TFileName)

