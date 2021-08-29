"""
Created on Sun Jul 11 22:10:18 2021

@author: root
"""

from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import librosa
import pickle
import numpy.matlib
from matplotlib import gridspec

import librosa.feature as rmses
import scipy

from scipy.signal import hilbert, chirp
from sklearn.preprocessing import MinMaxScaler

from scipy.signal import find_peaks


import FMP.libfmp.b
import FMP.libfmp.c2
import FMP.libfmp.c3
import FMP.libfmp.c4
from scipy.io import savemat



import cv2
#%%
#%%

file_name = "song_names"
file_path=[]
for files in open(file_name, "r"):
    file_path.append(files.rstrip('\n'))


#%%%%

base_dir = './' #base path which contains audios in a separate folder called 'audios'
posterier_dir = os.path.join(base_dir, 'Dump_Dir', 'posterier_inputs')
if not os.path.exists(posterier_dir): os.makedirs(posterier_dir)
#%%
for i in range (len(file_path)):
    print(i)

#%%
    acf_file_name = os.path.join(base_dir, 'Dump_Dir', 'data_dump', file_path[i].replace('.wav','.pickle'))
    nf_file_name = os.path.join(base_dir, 'Dump_Dir', 'data_dump', file_path[i].replace('.wav','_SSM_peaks.pickle'))

#%%
    '''Load Rhythmogram and onsets'''
    print('Load Rhythmogram and onsets')
    with open(acf_file_name, 'rb') as f:
        ODF, fs_ODF, log_ACF, sal, fs_ACF, onset_peaks, T_coef, onset_peaks_sec = pickle.load(f)
        f.close()
    log_ACF=log_ACF.T

#%%
    ''' Compute time and lag axes '''
    print('Compute time and lag axes')
    t_ACF = np.arange(len(log_ACF.T)) / (fs_ACF)
    f_ACF = np.arange(len(log_ACF)) / (fs_ODF)


#%%
    '''Load Novelty Function and peaks '''
    print('Load Novelty Function and peaks')
    with open(nf_file_name, 'rb') as f2:
        footes_nov, avg_std_1d, nov_fn_F_std, seg_peaks, seg_peaks_sec, properties = pickle.load(f2)
        f2.close()

#%%
    ACF=log_ACF
    ACF=ACF/np.max(ACF)
    ACF[0,:]=nov_fn_F_std
    scaler = MinMaxScaler()
    scaler.fit(ACF)
    ACF_norm=scaler.transform(ACF)

    #%%

    mdic = {"ACF_norm": ACF_norm, "label": "experiment", "f_ACF": f_ACF, "t_ACF": t_ACF}

    savemat(os.path.join(posterier_dir, file_path[i].replace('.wav','.mat')), mdic)
#%%
    print('Saving')
    with open(os.path.join(posterier_dir, file_path[i].replace('.wav','_post_input.pickle')), 'wb') as f:
        pickle.dump([ACF_norm, f_ACF, t_ACF], f)
    f.close()

