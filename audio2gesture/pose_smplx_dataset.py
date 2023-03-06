import os
import glob
from tqdm import tqdm
import pickle
import numpy as np

import pandas as pd

import torch
from torch.utils.data import Dataset

class VOCA_Dataset(Dataset):
    def __init__(self, subject_list):
        self.subject_list = sorted(subject_list)
        self.frames = 128

        if len(self.subject_list) > 0:
            print ('Loading the audio and mesh data...')
            self.dataset_audio = []
            self.dataset_audio_mfcc = []
            self.dataset_exp_param = []
            self.dataset_joints_param = []
            self.dataset_handl_param = []
            self.dataset_handr_param = []
            self.dataset_next_idx = []
            base =0
            for subject in tqdm(subject_list):
                audio_path = os.path.join('traindata/audio', subject)
                mesh_param_path = os.path.join('traindata/params', subject)
                audio_path_list = sorted(glob.glob(os.path.join(audio_path, '*.pkl')))
                mesh_param_path_list = sorted(glob.glob(os.path.join(mesh_param_path, '*.npz')))
                print ('\tPreparing ' + subject) 

                for audio_path, param_path in zip(audio_path_list, mesh_param_path_list):
                    audio_name = audio_path.split('/')[-1].replace('.pkl', '')
                    param_name = param_path.split('/')[-1].replace('.npz', '')
                    try:
                        assert audio_name == param_name
                    except:
                        print (audio_name, param_name)
                        
                    audio = pickle.load(open(audio_path, 'rb'), encoding=' iso-8859-1')

                    params = np.load(open(param_path, 'rb'))
                    pose = params['body_pose']
                    body_pose_embedding = params['body_pose_embedding']
                    jaw_pose = params['jaw_pose']
                    face = params['expression']
                    handl = params['left_hand_pose']
                    handr = params['right_hand_pose']
                    camera = params['camera_translation']


                    meancamera = np.mean(camera,axis=0)
                    stdcamera = np.std(camera,axis=0)

                    camera = (camera - meancamera)/(stdcamera+0.01)
                    
                    min_num_frame = min(pose.shape[0], audio.shape[0], 8200)

                    pose = pose[0:min_num_frame,:]
                    body_pose_embedding = body_pose_embedding[0:min_num_frame,:]
                    jaw_pose = jaw_pose[0:min_num_frame,:]
                    face = face[0:min_num_frame,:]
                    handl = handl[0:min_num_frame,:]
                    handr = handr[0:min_num_frame,:]
                    camera = camera[0:min_num_frame,:]
                    audio = audio[0:min_num_frame,:,:]

                    self.dataset_audio.append(audio)
                    self.dataset_exp_param.append(np.concatenate((body_pose_embedding, camera,  handl, handr, face, jaw_pose),axis=1))
                    self.dataset_next_idx += list(np.arange(0, min_num_frame-self.frames, 1) + base)
                    base+= min_num_frame


            self.dataset_audio = torch.Tensor(np.concatenate(self.dataset_audio))
            self.dataset_exp_param = torch.Tensor(np.concatenate(self.dataset_exp_param))

            self.dataset_next_idx = torch.Tensor(np.array(self.dataset_next_idx)).to(torch.long)

            assert self.dataset_audio.shape[0] == self.dataset_exp_param.shape[0] 
            self.length = len(self.dataset_next_idx)

    def __len__(self):
        return len(self.dataset_next_idx)

    def __getitem__(self, idx):

        audio = self.dataset_audio[self.dataset_next_idx[idx]:self.dataset_next_idx[idx]+self.frames]

        rand_idx = np.random.randint(self.dataset_audio.shape[0]-129)

        rand_audio = self.dataset_audio[rand_idx:rand_idx+self.frames]
        rand_exp_param = self.dataset_exp_param[rand_idx:rand_idx+self.frames]

        return audio, \
               self.dataset_exp_param[self.dataset_next_idx[idx]:self.dataset_next_idx[idx]+self.frames], \
                rand_audio, rand_exp_param

