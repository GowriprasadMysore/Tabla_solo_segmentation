from utils import *
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

###############################################################################
winsize_acf = 4  # 5 seconds window for Vilambit & 3 s for Drut
lagsize_acf = 2    # ACF calculated upto 1.5 seconds for drut 
hopsize_acf = 0.5  # successive ACF frames are shifted by 0.5 s
hopsize_stft = 0.005  # for a hop of 0.005 s
fs = int(1/hopsize_stft)

base_dir = './' #base path which contains audios in a separate folder called 'audios'
audio_dir = os.path.join(base_dir, 'dump_dir', 'audios')
acf_dir = os.path.join(base_dir, 'dump_dir', 'acf_data_%.2fsframe'%winsize_acf)
img_dir = os.path.join(base_dir, 'dump_dir', 'acf_plots_%.2fsframe'%winsize_acf)

tempo_save_dir = os.path.join(base_dir, 'dump_dir', 'acf_data_%.2fsframe'%winsize_acf)

data_dump_dir = os.path.join(base_dir, 'dump_dir', 'data_dump_dir%.2fsframe'%winsize_acf)
if not os.path.exists(tempo_save_dir): os.makedirs(tempo_save_dir)
#############################################################################

#songlist = os.listdir(audio_dir)

#%%
for file_path in open(file_name, "r"):
    file_path = file_path.rstrip('\n')
#    print('Audio: %s'%file_path)

    fields = file_path.rstrip().split("/")
    songname=fields[-1]
    print('Audio: %s'%songname)
    
  
    total_time = librosa.get_duration(filename=file_path, sr=16000)
    time_axis = np.arange(0, total_time, hopsize_acf)

	## Supply ground truth section boundaries if known
    GT_boundaries=[]
	
    log_ACF = np.load(os.path.join(acf_dir, songname.replace('.wav','_logACF.npy')))
    time_axis = time_axis[:len(log_ACF)]
    total_time = time_axis[-1]
    print('Loaded data')
    print('Calculate metric tempo')           
    ## Calculate metric tempo
    [window, tempo_candidates, tempo_period] = tempo_period_comb_filter(log_ACF, fs)
    metric_tempo_period_vit = viterbi_tempo_rhythm(tempo_candidates, fs, 1.5)
    metric_tempo = 60/metric_tempo_period_vit

    print('tempo computed')           
	## Display metric tempo period on rhythmogram
    display_ACF(log_ACF, total_time, lagsize_acf, GT_boundaries, take_log=0, heading='Rhythmogram with metric tempo period')
    plt.plot(time_axis, metric_tempo_period_vit, marker='.', ms=10, ls='None', color='r')
    plt.show()
    plt.savefig(os.path.join(img_dir, songname.replace('.wav','metric_tempi.png')))
    plt.clf()
    print('saved metric tempo') 
    print('Calculate surface tempo') 
    ## Calculate surface tempo
    [window, tempo_candidates, tempo_period] = tempo_period_comb_filter_surf(log_ACF, fs)
    surface_tempo_period_vit = viterbi_tempo_rhythm(tempo_candidates, fs, 0.1)
    surface_tempo = 60/surface_tempo_period_vit
    print('tempo computed')           
	## Display surface tempo period on rhythmogram
    display_ACF(log_ACF, total_time, lagsize_acf, GT_boundaries, take_log=0, heading='Rhythmogram with surface tempo period')
    plt.plot(time_axis, surface_tempo_period_vit, marker='.', ms=10, ls='None', color='b')
    plt.show()
    plt.savefig(os.path.join(img_dir, songname.replace('.wav','surface_tempi.png')))
    plt.clf()
    print('saved surface tempo') 
	## Display metric and surface tempi in BPM
    display_met_surf_tempo(time_axis, total_time, metric_tempo_period_vit, surface_tempo_period_vit, GT_boundaries, heading='Metric Tempo and Surface Tempo')
    plt.show()
    plt.savefig(os.path.join(img_dir, songname.replace('.wav','tempi.png')))
    plt.clf()
    plt.close('all')
	
    tempi_pred = np.hstack((np.atleast_2d(60/metric_tempo_period_vit).T, np.atleast_2d(60/surface_tempo_period_vit).T))
    
    with open(os.path.join(data_dump_dir, songname.replace('.wav','_tempo_params.pickle')), 'wb') as f:
        pickle.dump([tempi_pred, metric_tempo_period_vit, surface_tempo_period_vit], f)
    f.close()
    print('Saving')

