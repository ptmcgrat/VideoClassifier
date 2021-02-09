import os, pdb
import pandas as pd
import numpy as np
import torch.utils.data as data
from PIL import Image
#from SpatialTransforms import 

class JPGLoader(data.Dataset):
    def __init__(self, data_folder, videofiles_projectID_dt, xy_crop, t_crop, t_interval, augment = False, projectMeans = True):
        # videofiles_projectID_dt is a pandas dataframe with column VideoFile, column ProjectID, 

        self.data_folder = data_folder
        self.dt = videofiles_projectID_dt
        self.xy_crop = xy_crop
        self.t_crop = t_crop
        self.t_interval = t_interval
        self.augment = augment

        self.dt['Mean'] = np.nan
        self.dt['Std'] = np.nan

    def _convertMP4s_to_Jpegs(self):

        for mp4file in [x for x in dt.VideoFile if '.mp4' in x]:
            # Convert mp4 to jpg
            cmd = ['ffmpeg','-i', self.data_folder + mp4file, self.data_folder + mp4file.replace('.mp4','') + '/image_%05d.jpg']
            subprocess.run(cmd)

    def _calculateProjectIDstats(self):

        # Calculate mean and std for single image from each videofile
        for index, row in dt.iterrows():
            image_path = os.path.join(self.data_folder + row.Videofile.replace('.mp4',''), 'image_00001.jpg')
            with open(image_path, 'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('RGB')

                    means = np.mean(img, axis = (0,1))
                    dt.at[index,'MeanR'] = np.mean(img, axis = (0,1))[0]
                    dt.at[index,'MeanG'] = np.mean(img, axis = (0,1))[1]
                    dt.at[index,'MeanB'] = np.mean(img, axis = (0,1))[2]

                    stds = np.std(img, axis = (0,1))
                    dt.at[index,'StdR'] = np.std(img, axis = (0,1))[0]
                    dt.at[index,'StdG'] = np.std(img, axis = (0,1))[1]
                    dt.at[index,'StdB'] = np.std(img, axis = (0,1))[2]

        # Create dictionary of projectIDs with mean and stdev data
        self.project_dict = {}
        for index,row in dt.groupby('MeanID').mean().iterrows():
            self.project_dict[index] = [np.array(row[0:3]),np.array(row[4:7])]


    def __getitem__(self, index):
        pass

dt = pd.read_csv('Test.csv')
dt['MeanR'] = np.nan
dt['MeanG'] = np.nan
dt['MeanB'] = np.nan
dt['StdR'] = np.nan
dt['StdG'] = np.nan
dt['StdB'] = np.nan

image_path = os.path.join('0001_vid__3747__505__4955__606__221', 'image_00001.jpg')
with open(image_path, 'rb') as f:
    with Image.open(f) as img:
        img = img.convert('RGB')

for index, row in dt.iterrows():
    means = np.mean(img, axis = (0,1))
    dt.at[index,'MeanR'] = np.mean(img, axis = (0,1))[0]
    dt.at[index,'MeanG'] = np.mean(img, axis = (0,1))[1]
    dt.at[index,'MeanB'] = np.mean(img, axis = (0,1))[2]

    stds = np.std(img, axis = (0,1))
    dt.at[index,'StdR'] = np.std(img, axis = (0,1))[0]
    dt.at[index,'StdG'] = np.std(img, axis = (0,1))[1]
    dt.at[index,'StdB'] = np.std(img, axis = (0,1))[2]

project_dt = dt.groupby('MeanID').mean()
project_dict = {}
for index,row in project_dt.iterrows():
    project_dict[index] = [np.array(row[0:3]),np.array(row[4:7])]

pdb.set_trace()
