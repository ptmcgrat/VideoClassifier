import os, pdb, subprocess
import pandas as pd
import numpy as np
import torch.utils.data as data
from PIL import Image

from Utils.VideoTransforms import TransformJPEGs
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

        self.dt = self.dt.reset_index()

        self.dt['VideoExists'] = False

        self.dt['MeanR'] = np.nan
        self.dt['MeanG'] = np.nan
        self.dt['MeanB'] = np.nan

        self.dt['StdR'] = np.nan
        self.dt['StdG'] = np.nan
        self.dt['StdB'] = np.nan

        self._convertMP4s_to_Jpegs()
        self._calculateProjectIDstats()

        self.transforms = {}

        for projectID, values in self.project_dict.items():
            self.transforms[projectID] = TransformJPEGs(self.xy_crop, self.t_crop, self.t_interval, augment = self.augment, mean = values[0], std = values[1])

        self.target_transform = sorted(set(self.dt.Label))

        test = self.__getitem__(1)

    def _convertMP4s_to_Jpegs(self):

        print('Converting mp4 files to jpgs')
        for i,mp4file in enumerate([x for x in self.dt.VideoFile if '.mp4' in x]):
            if not os.path.exists(self.data_folder + mp4file):
                continue
 
            if i%500 == 0:
                print('Converted ' + str(i) + ' videos of ' + str(len([x for x in self.dt.VideoFile if '.mp4' in x])))
            # Convert mp4 to jpg
            try:    
                os.mkdir(self.data_folder + mp4file.replace('.mp4','') )
            except:
                continue
            cmd = ['ffmpeg','-i', self.data_folder + mp4file, self.data_folder + mp4file.replace('.mp4','') + '/image_%05d.jpg']
            output = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(self.data_folder + mp4file.replace('.mp4','') + '/image_00001.jpg'):
                self.dt.at[i,'VideoExists'] = True


        print('Cant find ' + str(len(self.dt[self.dt.VideoExists == False])) + ' videos.')
        self.dt = self.dt[self.dt.VideoExists == True]
        self.dt = self.dt.reset_index()

    def _calculateProjectIDstats(self):
        print('Calculating project means')

        # Calculate mean and std for single image from each videofile
        for index, row in self.dt.iterrows():
            image_path = os.path.join(self.data_folder + row.VideoFile.replace('.mp4',''), 'image_00001.jpg')
            try:
                with open(image_path, 'rb') as f:
                    with Image.open(f) as img:
                        img = img.convert('RGB')

                        means = np.mean(img, axis = (0,1))
                        self.dt.at[index,'MeanR'] = np.mean(img, axis = (0,1))[0]
                        self.dt.at[index,'MeanG'] = np.mean(img, axis = (0,1))[1]
                        self.dt.at[index,'MeanB'] = np.mean(img, axis = (0,1))[2]

                        stds = np.std(img, axis = (0,1))
                        self.dt.at[index,'StdR'] = np.std(img, axis = (0,1))[0]
                        self.dt.at[index,'StdG'] = np.std(img, axis = (0,1))[1]
                        self.dt.at[index,'StdB'] = np.std(img, axis = (0,1))[2]
            except:
                print('Cant find ' + image_path)
        # Create dictionary of projectIDs with mean and stdev data
        self.project_dict = {}
        for index,row in self.dt.groupby('ProjectID').mean()[['MeanR','MeanG','MeanB','StdR','StdG','StdB']].iterrows():
            self.project_dict[index] = [np.array(row[0:3]),np.array(row[3:6])]


    def __len__(self):
        return len([x for x in self.dt.VideoFile if '.mp4' in x])

    def __getitem__(self, index):
        videofile = self.dt.iloc[index].VideoFile
        projectID = self.dt.iloc[index].ProjectID
        video_path = os.path.join(self.data_folder, videofile.replace('.mp4',''))
        label = self.dt.iloc[index].Label


        num_images = len([x for x in os.listdir(video_path) if '.jpg' in x])

        video = []
        for i in range(num_images):
            with open(os.path.join(video_path, 'image_{:05d}.jpg'.format(index)), 'rb') as f:
                with Image.open(f) as img:
                    video.append(img.convert('RGB'))
        return self.transforms[projectID](video), self.target_transform.index(label)

