import os
import numpy as np
import torch
import argparse

import scipy
import copy
from scipy.io import wavfile

from torch.utils.data import DataLoader

import pickle
from tqdm import tqdm

from pose_smplx_dataset import *
from    model import TposeGANsmplx

import glob
from os.path import join, exists, abspath, dirname
num_params = 72
out_path = '../examples/test-result'


if not os.path.exists(out_path):
	os.makedirs(out_path)
subject = 'A'
training_set = VOCA_Dataset([subject])
rand_idx = np.random.randint(training_set.dataset_audio.shape[0]-129)
rand_exp_param = training_set.dataset_exp_param[rand_idx]
# print(rand_exp_param)
audio_list = glob.glob('../examples/audio_preprocessed/A_test.pkl')
for audio_path in audio_list:
    # print (audio_path)
    processed_audio = pickle.load(open(audio_path, 'rb'), encoding=' iso-8859-1')
    processed_audio = processed_audio[:,:,:]

    modelgen = TposeGANsmplx().cuda()
    train_path = './checkpoint/Gen-100-0.004101611138283477.mdl'

    modelgen.load_state_dict(torch.load(train_path))

    modelgen.eval()

    processed_audio = torch.Tensor(processed_audio)
    audioname = audio_path.split('/')[-1].replace('.pkl', '')

    faceparams = np.zeros((processed_audio.shape[0], num_params), float)

    frames_out_path = os.path.join(out_path, 'A_test.npz')
    firstpose = torch.zeros([1,num_params],dtype=torch.float32).unsqueeze(0)
    firstpose[0]=torch.Tensor(rand_exp_param)
    initial_num = 35
    with torch.no_grad():
        for repeat in range(1):
            for i in range(0,processed_audio.shape[0]-127, 127):
            
                audio = processed_audio[i:i+128,:,:].unsqueeze(0).cuda()

                _faceparam = modelgen(audio,firstpose.cuda()[:,:,:initial_num])


                firstpose = _faceparam[:,127:128,:initial_num]
                faceparams[i:i+128,:] = _faceparam[0,:,:].cpu().numpy()

                # last audio sequence
                if i+127 >= processed_audio.shape[0]-127:
                    j = processed_audio.shape[0]-128
                    audio = processed_audio[j:j+128,:,:].unsqueeze(0).cuda()
                    firstpose = _faceparam[:,j-i:j-i+1,:initial_num]
                    _faceparam = modelgen(audio,firstpose.cuda()[:,:,:initial_num])
                    faceparams[j:j+128,:] = _faceparam[0,:,:].cpu().numpy()

        np.savez(frames_out_path, face = faceparams)