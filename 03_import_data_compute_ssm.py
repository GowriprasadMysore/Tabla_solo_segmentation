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

from scipy.signal import find_peaks


import FMP.libfmp.b
import FMP.libfmp.c2
import FMP.libfmp.c3
import FMP.libfmp.c4



import cv2
#%%

file_name = "song_names"
file_path=[]
for files in open(file_name, "r"):
    file_path.append(files.rstrip('\n'))


#%%%%

base_dir = './' #base path which contains audios in a separate folder called 'audios'
data_dump_dir = os.path.join(base_dir, 'Dump_Dir', 'data_dump')
img_dir = os.path.join(base_dir, 'Dump_Dir', 'result_plots')

if not os.path.exists(img_dir): os.makedirs(img_dir)
if not os.path.exists(data_dump_dir): os.makedirs(data_dump_dir)

#%%
I=[16]
for i in I:

#for i in range (len(file_path)):
    print(i)

#%%
    acf_file_name = os.path.join(base_dir, 'Dump_Dir', 'data_dump', file_path[i].replace('.wav','.pickle'))
    sd_file_name = os.path.join(base_dir, 'Dump_Dir', 'data_dump', file_path[i].replace('.wav','_SD_params.pickle'))
    nf_file_name = os.path.join(base_dir, 'Dump_Dir', 'data_dump', file_path[i].replace('.wav','_SSM_peaks.pickle'))
    img_name = os.path.join(base_dir, 'Dump_Dir', 'result_plots', file_path[i].replace('.wav','_nov_fn.png'))


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
    '''Load stroke density and rmse '''
    print('Load stroke density and rmse')
    with open(sd_file_name, 'rb') as f1:
        stroke_density, stroke_density_avg, t_sd, local_average_rmse, t_rmse, Rh_flux_HE_avg = pickle.load(f1)
        f1.close()


#%%
    stroke_density_avg=np.array(stroke_density_avg)
    stroke_density_avg_diff=np.diff(np.array(stroke_density_avg))
    stroke_density_avg_diff_smooth = utils.anchor(stroke_density_avg_diff, 0.85)

    analytic_signal = hilbert(stroke_density_avg_diff_smooth)
    avg_std_1d = np.abs(analytic_signal)
    avg_std_1d=np.concatenate((avg_std_1d, np.zeros((1))), axis=0)


#%%
    S = utils.compute_sm_dot(log_ACF,log_ACF)
    L_kernel = 50
    nov = utils.compute_novelty_ssm(S, L=L_kernel, exclude=True)
    footes_nov = utils.anchor(nov, 0.85)

    footes_nov=np.array(footes_nov)
    footes_nov=footes_nov/np.max(footes_nov)


#%%

    nov_fn=footes_nov+avg_std_1d
    nov_fn_F_std = utils.anchor(nov_fn, 0.85)
    nov_fn_F_std=np.array(nov_fn_F_std)

#%%

    #peaks, _ = find_peaks(avg_std_1d, distance=50)
    seg_peaks, properties = find_peaks(nov_fn_F_std, prominence=0.3,distance=35,width=5)
    #nov_fn_F_std=nov_fn_F_std/np.max(nov_fn_F_std)

    seg_peaks_sec=seg_peaks/2

#%%
    # create a figure
    print('create a figure')
    fig = plt.figure(figsize=(20,14))
    spec = gridspec.GridSpec(ncols=1, nrows=5, hspace=0.4, height_ratios=[1.5, 0.7, 1, 1, 1])

    ax0 = fig.add_subplot(spec[0])
    ax0.pcolormesh(t_ACF/60, f_ACF, log_ACF)
    ax0.vlines(seg_peaks/120, ymin = max(f_ACF)/4, ymax = max(f_ACF),linewidth=2, colors = 'yellow', linestyles='dashdot')
    ax0.set_title('Rhythmogram',fontsize= 30)
    ax0.set_ylabel('Autocorelation Lag in sec',fontsize= 16)
    ax0.set_xticks(np.arange(0, max(t_ACF)/60, 1))

    ax1 = fig.add_subplot(spec[1])
    ax1.plot(t_sd/60,stroke_density_avg/4,linewidth=4.0)
    ax1.vlines(seg_peaks/120, ymin = 0, ymax = max(stroke_density_avg)/4,linewidth=2.0, colors = 'red')
    ax1.set_xlim(0.0, t_sd[-1]/60)
    ax1.set_title('Avg Stroke density',fontsize= 20)


    ax2 = fig.add_subplot(spec[2])
    ax2.plot(t_ACF/60,nov_fn_F_std,linewidth=3.0)
    ax2.plot(seg_peaks/120, nov_fn_F_std[seg_peaks], "x", markersize=10)
    ax2.vlines(seg_peaks/120, ymin = 0, ymax = max(nov_fn_F_std),linewidth=0.5, linestyles='dashdot')
    ax2.set_xlim(0.0, t_ACF[-1]/60)
    ax2.set_title('Novelty function' ,fontsize= 20)


    ax3 = fig.add_subplot(spec[3])
    ax3.plot(t_sd/60,avg_std_1d,linewidth=3.0)
    #ax2.plot(seg_peaks/2, x[seg_peaks], "x", markersize=12)
    ax3.vlines(seg_peaks/120, ymin = 0, ymax = max(avg_std_1d),linewidth=0.5, linestyles='dashdot')
    ax3.set_xlim(0.0, t_sd[-1]/60)
    ax3.set_title('Avg Stroke density 1st Diff ',fontsize= 20)
    #ax2.set_ylabel('Rhythmogram Flux',fontsize= 16)


    ax4 = fig.add_subplot(spec[4])
    #ax3.plot(t_rmse,local_average_rmse,linewidth=4.0)
    ax4.plot(t_ACF/60,footes_nov,linewidth=4.0)
    ax4.vlines(seg_peaks/120, ymin = 0, ymax = max(footes_nov),linewidth=0.5, linestyles='dashdot')
    ax4.set_xlim(0.0, t_ACF[-1]/60)
    ax4.set_title('Foote-s Novelty function' ,fontsize= 20)
    ax4.set_xlabel('Time [mins]',fontsize= 20)


    plt.show()

#%%
#
#    plt.savefig(img_name)
#    plt.clf()
#    plt.close('all')
#
##%%
#    print('Saving')
#    with open(nf_file_name, 'wb') as f:
#        pickle.dump([footes_nov, avg_std_1d, nov_fn_F_std, seg_peaks, seg_peaks_sec, properties], f)
#    f.close()
#
