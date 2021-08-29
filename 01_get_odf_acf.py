import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import librosa
import pickle
import csv
import numpy as np
import pandas as pd
import os
import librosa
import numpy.matlib
#%%

file_name = "Concert_list"
file1 = open(file_name, 'r')
file_paths = file1.readlines()
file1.close()
#%%
#file_name=[]
#for line in file_paths:
#  fields = line.rstrip().split("/")
#  file_name.append(fields[-1])
#
#cwd = os.getcwd()  # Currect directory absolute path
#audio_dir = cwd+'/audio'

#%%
###############################################################################
winsize_stft = 0.04  # for a frame width of 0.04s
hopsize_stft = 0.005  # for a hop of 0.005 s
lagsize_acf = 2     # ACF calculated upto 3 seconds for Vil & 1.5s for Madhya & Drut
winsize_acf = 4  # 5 seconds window for Vilambit & 3s for Madhya & Drut
hopsize_acf = 0.5  # successive ACF frames are shifted by 0.5 s

base_dir='./' #base path which contains audios in a separate folder called 'audios'
#audio_dir = os.path.join(base_dir, 'audios')


#paths to save novelty curve data, ACF data, and rhythmogram plots
img_dir = os.path.join(base_dir, 'Dump_Dir', 'acf_plots_%.2fsframe'%winsize_acf)
data_dump = os.path.join(base_dir, 'Dump_Dir', 'data_dump')


if not os.path.exists(img_dir): os.makedirs(img_dir)
if not os.path.exists(data_dump): os.makedirs(data_dump)

print('Created Directories')

#############################################################################

#audios_list = os.listdir(audio_dir)

#%%
for file_path in open(file_name, "r"):
    file_path = file_path.rstrip('\n')
#    print('Audio: %s'%file_path)

    fields = file_path.rstrip().split("/")
    songname=fields[-1]
    print('Audio: %s'%songname)

    audio, fs_audio = librosa.load(file_path, sr=16000)
    print('Loaded audio')
    ODF, fs_ODF = utils.compute_novelty_spectrum(audio, Fs=fs_audio, N=int(2**np.ceil(np.log2(winsize_stft*fs_audio))), W=int(winsize_stft*fs_audio), H=int(hopsize_stft*fs_audio), gamma=100, M=10, norm=1)
    print('Computed ODF')
    
    peaks, T_coef, peaks_sec = utils.find_onsets(ODF, fs_ODF, prominence=0.02)
    print('Computed Onsets')
    
    ACF, DFT, sal = utils.ACF_DFT_sal(ODF, lagsize_acf, winsize_acf, hopsize_acf, fs_ODF)
    log_ACF = 10*np.log10(1 + 100*ACF)
    fs_ACF=1/hopsize_acf    
    print('Computed ACF')
    
    fig, ax = plt.subplots(1,1);
    #ax.imshow(log_ACF.T, origin='lower', aspect='auto', cmap='gray')
    ax.imshow(log_ACF.T, origin='lower', aspect='auto')
    ax.set_title('Rhythmogram', fontsize=12)
    ax.tick_params(axis='both', labelsize=8)

    with open(os.path.join(data_dump, songname.replace('.wav','.pickle')), 'wb') as f:
        pickle.dump([ODF, fs_ODF, log_ACF, sal, fs_ACF, peaks, T_coef, peaks_sec], f)
    f.close()
    print('Saving')

    plt.savefig(os.path.join(img_dir, songname.replace('.wav','.png')))
    plt.clf()
    plt.close('all')

