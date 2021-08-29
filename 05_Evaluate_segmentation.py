from utils import *
#%%
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
#%%

import FMP.libfmp.b
import FMP.libfmp.c2
import FMP.libfmp.c3
import FMP.libfmp.c4

from scipy.fftpack import fft, fftshift
import scipy.signal as signal

import cv2

import mir_eval

#%%

file_name = "song_names"
file_path=[]
for files in open(file_name, "r"):
    file_path.append(files.rstrip('\n'))


#%%%%

base_dir = './' #base path which contains audios in a separate folder called 'audios'
img_dir = os.path.join(base_dir, 'Dump_Dir', 'Compare_plots')

if not os.path.exists(img_dir): os.makedirs(img_dir)

#%%
#I=[0,2,4,8,64,69,79]
I=[16]

for i in I:
#for i in range (len(file_path)):
    print(i)

#%%
    acf_file_name = os.path.join(base_dir, 'Dump_Dir', 'data_dump', file_path[i].replace('.wav','.pickle'))
    sd_file_name = os.path.join(base_dir, 'Dump_Dir', 'data_dump', file_path[i].replace('.wav','_SD_params.pickle'))
    nf_file_name = os.path.join(base_dir, 'Dump_Dir', 'data_dump', file_path[i].replace('.wav','_SSM_peaks.pickle'))
    img_name = os.path.join(img_dir, file_path[i].replace('.wav','_cmpr.png'))

    Ann_file = os.path.join(base_dir,'Annotations', file_path[i].replace('.wav','.csv'))

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
    '''Load stroke density and rmse '''
    print('Load stroke density and rmse')
    with open(sd_file_name, 'rb') as f1:
        stroke_density, stroke_density_avg, t_sd, local_average_rmse, t_rmse, Rh_flux_HE_avg = pickle.load(f1)
        f1.close()
    stroke_density_avg=np.array(stroke_density_avg)


#%%
    '''Load Novelty Function and peaks '''
    print('Load Novelty Function and peaks')
    with open(nf_file_name, 'rb') as f2:
        footes_nov, avg_std_1d, nov_fn_F_std, seg_peaks, seg_peaks_sec, properties = pickle.load(f2)
        f2.close()


#%%
    """ Load Ground Truth Annotation """
    print('Load Ground Truth Annotation')
    ann, color_ann = FMP.libfmp.c4.read_structure_annotation(Ann_file, fn_ann_color=Ann_file)
    end_insts = sorted(set([end for start, end, label in ann]))
    gt_ann=np.array(end_insts)
    gt_ann=gt_ann[:-1]

    gt_ann=np.round(gt_ann)
    gt_ann = gt_ann.astype(int)

    #%%
    """ Evaluate Performance and generate gaussian one hot vector"""

    scores=mir_eval.onset.evaluate(gt_ann, seg_peaks_sec, window=40)
    gauss_sdf=utils.generate_gaussian_output_vector(gt_ann, t_ACF, std=5, ln=11)

    #%%
    title=("Rhythmogram, P=%f"%scores['Precision'] + ", R=%f"%scores['Recall'] + ", F=%f" %scores['F-measure'])
    # create a figure
    print('create a figure')
    fig = plt.figure(figsize=(20,14))
    spec = gridspec.GridSpec(ncols=1, nrows=4, hspace=0.4, height_ratios=[1.5, 0.7, 1, 1])

    ax0 = fig.add_subplot(spec[0])
    ax0.pcolormesh(t_ACF, f_ACF, log_ACF)
    ax0.vlines(seg_peaks_sec, ymin = max(f_ACF)/2, ymax = max(f_ACF),linewidth=2, colors = 'red', linestyles='dashdot', label ='Predicted')
    ax0.vlines(gt_ann, ymin = max(f_ACF)/4, ymax = 3*max(f_ACF)/4,linewidth=2, colors = 'yellow', label ='Ground-Truth')
    ax0.set_title(title,fontsize= 30)
    ax0.set_ylabel('Autocorelation Lag in sec',fontsize= 16)
    ax0.legend(seg_peaks_sec, ['Ground-Truth', 'line B'],loc='upper right', frameon=False)

    ax1 = fig.add_subplot(spec[1])
    ax1.plot(t_sd,stroke_density_avg/4,linewidth=4.0)
    ax1.vlines(seg_peaks_sec, ymin = 0, ymax = max(stroke_density_avg)/4,linewidth=2.0, colors = 'red')
    ax1.set_xlim(0.0, t_sd[-1])
    ax1.set_title('Avg Stroke density',fontsize= 20)


    ax2 = fig.add_subplot(spec[2])
    ax2.plot(t_ACF,nov_fn_F_std,linewidth=3.0)
    ax2.plot(seg_peaks_sec, nov_fn_F_std[seg_peaks], "x", markersize=10)
    ax2.vlines(seg_peaks_sec, ymin = 0, ymax = max(nov_fn_F_std),linewidth=0.5, linestyles='dashdot')
    ax2.set_xlim(0.0, t_ACF[-1])
    ax2.set_title('Novelty function' ,fontsize= 20)




    ax3 = fig.add_subplot(spec[3])
    ax3.plot(t_sd,gauss_sdf,linewidth=3.0)
    #ax2.plot(peaks/2, x[peaks], "x", markersize=12)
    ax3.vlines(seg_peaks_sec, ymin = 0, ymax = 1.5, linewidth=0.5, linestyles='dashdot')
    #ax3.vlines(gt_ann, ymin = 0, ymax = max(amplitude_envelope),linewidth=0.5, linestyles='dashdot')
    ax3.set_xlim(0.0, t_sd[-1])
    ax3.set_title('Gaussian one hot vector',fontsize= 20)
    #ax2.set_ylabel('Rhythmogram Flux',fontsize= 16)
    ax3.set_xlabel('Time [sec]',fontsize= 20)


    plt.show()
    #%%

    plt.savefig(img_name)
    plt.clf()
    plt.close('all')
