from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import librosa
import pickle
import csv
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

import librosa.feature as rmses
import scipy

from scipy.signal import hilbert, chirp

import cv2
#%%

file_name = "Concert_list"
file1 = open(file_name, 'r')
file_paths = file1.readlines()
file1.close()

###############################################################################
'''Parameter Initialization'''
winsize_acf = 4  # 5 seconds window for Vilambit & 3 s for Drut
lagsize_acf = 2    # ACF calculated upto 1.5 seconds for drut
hopsize_acf = 0.5  # successive ACF frames are shifted by 0.5 s
hopsize_stft = 0.005  # for a hop of 0.005 s


#%%%%
base_dir = './' #base path which contains audios in a separate folder called 'audios'
data_dump_dir = os.path.join(base_dir, 'Dump_Dir', 'data_dump')

#############################################################################

#%%
for file_path in open(file_name, "r"):
    file_path = file_path.rstrip('\n')
#    print('Audio: %s'%file_path)

    fields = file_path.rstrip().split("/")
    songname=fields[-1]
    print('Audio: %s'%songname)

    dump_file = os.path.join(data_dump_dir,songname.replace('.wav','.pickle'))

#%%
    '''Load Audio'''
    print('Load Audio')
    audio, fs_audio = librosa.load(file_path, sr=16000)

    #%%
    '''Load Rhythmogram and onsets'''
    print('Load Rhythmogram and onsets')
    with open(dump_file, 'rb') as f:
        ODF, fs_ODF, log_ACF, sal, fs_ACF, peaks, T_coef, peaks_sec = pickle.load(f)
        f.close()
    log_ACF=log_ACF.T

    #%%
    ''' Compute time and lag axes '''
    print('Compute time and lag axes')
    t_ACF = np.arange(len(log_ACF.T)) / (fs_ACF)
    f_ACF = np.arange(len(log_ACF)) / (fs_ODF)
    
    
   # '''Perform SobelX operation'''
    #sobelx = cv2.Sobel(log_ACF,cv2.CV_64F,1,0,ksize=13)  # x
    
    
    #%%
    
    '''Rhythmogram Flux Computation'''
    print('Rhythmogram Flux Computation')
    Rh_flux=np.sum(np.diff(log_ACF),axis=0)
    Rh_flux_H = hilbert(Rh_flux)
    Rh_flux_HE = np.abs(Rh_flux_H)
    Rh_flux_HE=np.pad(Rh_flux_HE, (0, 1), 'constant')
    Rh_flux_HE_avg = utils.anchor(Rh_flux_HE, 0.95)


    #%%
    ''' Compute Short term energy '''
    print('Compute Short term energy')
    hop_length = fs_audio*hopsize_acf
    frame_length = fs_audio*winsize_acf
    rmse = rmses.rms(audio, frame_length=int(frame_length), hop_length=int(hop_length), center=True)
    local_average_rmse = utils.anchor(rmse.T, 0.95)
    t_rmse = np.arange(len(rmse.T)) / (fs_ACF)

#%%
    ''' Compute Stroke Density'''
    print('Compute Stroke Density')
    onsets=np.zeros(len(ODF))
    onsets[peaks]=1
    n_ACF_frame = int(winsize_acf*fs_ODF)
    n_ACF_hop = int(hopsize_acf*fs_ODF)
    onsets_chunks = utils.subsequences(onsets, n_ACF_frame, n_ACF_hop)
    stroke_density=np.sum(onsets_chunks,axis=1)
    stroke_density_avg = utils.anchor(stroke_density, 0.96)
    t_sd = np.arange(len(stroke_density)) / (fs_ACF)

#%%
    print('Saving')
    with open(os.path.join(data_dump_dir, songname.replace('.wav','_SD_params.pickle')), 'wb') as f:
        pickle.dump([stroke_density, stroke_density_avg, t_sd, local_average_rmse, t_rmse, Rh_flux_HE_avg], f)
    f.close()

