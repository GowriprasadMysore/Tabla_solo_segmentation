import numpy as np
import scipy.signal as sig
import librosa
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # used for plotting the greyscale image of spectrogram and rhythmogram
import FMP.libfmp.b
import FMP.libfmp.c2
import FMP.libfmp.c3
import FMP.libfmp.c4

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

from scipy.fftpack import fft, fftshift
import scipy.signal as signal

import cv2


def compute_local_average(x, M, Fs=1):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb (Source: FMP Notebooks by Meinard Mueller

    Args:
        x: Signal
        M: Determines size (2M+1*Fs) of local average
        Fs: Sampling rate

    Returns:
        local_average: Local average signal
    """
    L = len(x)
    M = int(np.ceil(M * Fs))
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average

def compute_novelty_spectrum(x, Fs=1, N=1024, W=640, H=80, gamma=100, M=10, norm=1, band=[]):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb (Source: FMP Notebooks by Meinard Mueller

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        gamma: Parameter for logarithmic compression
        M: Size (frames) of local average
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=W, window='hanning')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
	
	#if frequency band provided:
    if len(band)!=0: 
        band = np.array(band)*(N/2+1)/Fs
        Y = Y[int(band[0]):int(band[1]),:]

    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm == 1:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature

def subsequences(signal, frame_length, hop_length):
    shape = (int(1 + (len(signal) - frame_length)/hop_length), frame_length)
    strides = (hop_length*signal.strides[0], signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

def ACF_DFT_sal(signal, t_ACF_lag, t_ACF_frame, t_ACF_hop, fs):
    n_ACF_lag = int(t_ACF_lag*fs)
    n_ACF_frame = int(t_ACF_frame*fs)
    n_ACF_hop = int(t_ACF_hop*fs)
    signal = subsequences(signal, n_ACF_frame, n_ACF_hop)
    ACF = np.zeros((len(signal), n_ACF_lag))
    for i in range(len(ACF)):
        ACF[i][0] = np.dot(signal[i], signal[i])
        for j in range(1, n_ACF_lag):
            ACF[i][j] = np.dot(signal[i][:-j], signal[i][j:])
    DFT = (abs(np.fft.rfft(signal)))
    sal = np.zeros(len(ACF))
    for i in range(len(ACF)):
        sal[i] = max(ACF[i])
    for i in range(len(ACF)):
        if max(ACF[i])!=0:
            ACF[i] = ACF[i]/max(ACF[i])
        if max(DFT[i])!=0:
            DFT[i] = DFT[i]/max(DFT[i])
    return (ACF, DFT, sal)    

def tempo_period_comb_filter(ACF, fs, norm=1):
    L = np.shape(ACF)[1]
    min_lag = 10
#    max_lag = L/2    #For madhya laya
    max_lag = L     # For Vil bandish
#    max_lag = 66    # For Drut gats
    N_peaks = 11    # 11 for Madhya laya & 9 for Drut gat  
    
 
    window = zeros((L, L))
    for j in range(min_lag, max_lag):
        C = j*arange(1, N_peaks)
        D = concatenate((C, C+1, C-1, C+2, C-2, C+3, C-3))
        D = D[D<L]
        norm_factor = len(D)
        if norm == 1:
            window[j][D] = 1.0/norm_factor
        else:
            window[j][D] = 1.0
            
    tempo_candidates = dot(ACF, transpose(window))
    
#    re_weight = zeros(L)
#    re_weight[min_lag:max_lag] = linspace(1, 0.5, max_lag-min_lag)
#    tempo_candidates = tempo_candidates*re_weight
    tempo_lag = argmax(tempo_candidates, axis=1)/float(fs)
    return (window, tempo_candidates, tempo_lag)
    
def tempo_period_comb_filter_surf(ACF, fs, norm=1):
    L = np.shape(ACF)[1]
    min_lag = 20
#    max_lag = L/2    #For madhya laya
    max_lag = L     # For Vil drut bandish
#    max_lag = 66    # For Drut gats
    N_peaks = 5    # 5 for Vil laya bandish; 9 for mdhya & drut bandish
    window = zeros((L, L))
    for j in range(min_lag, max_lag):
        C = j*arange(1, N_peaks)
        D = concatenate((C, C+1, C-1, C+2, C-2, C+3, C-3))
#        D = concatenate((C, C+1, C-1, C+2, C-2))
        D = D[D<L]
        norm_factor = len(D)
        if norm == 1:
            window[j][D] = 1.0/norm_factor
        else:
            window[j][D] = 1.0
            
    tempo_candidates = dot(ACF, transpose(window))
    
#    re_weight = zeros(L)
#    re_weight[min_lag:max_lag] = linspace(1, 0.5, max_lag-min_lag)
#    tempo_candidates = tempo_candidates*re_weight
    tempo_lag = argmax(tempo_candidates, axis=1)/float(fs)
    return (window, tempo_candidates, tempo_lag)
    
def viterbi_tempo_rhythm(tempo_candidates, fs, transition_penalty):
    T = np.shape(tempo_candidates)[0]
    L = np.shape(tempo_candidates)[1]

    p1=transition_penalty

    cost=ones((T,L//2))*1e8
    m=zeros((T,L//2))

#    cost[:,0]=1000
#    m[0][1]=argmax(tempo_candidates[0])
#    for j in range(1,L):
#        cost[1][j]=abs(60*fs/j-60*fs/m[0][1])/tempo_candidates[1][j]
#        m[1][j]=m[0][1]/fs

    for i in range(1,T):
        for j in range(1,L//2):
            cost[i][j]=cost[i-1][1]+p1*abs(60.0*fs/j-60.0*fs)-tempo_candidates[i][j]
            for k in range(2,L//2):
                if cost[i][j]>cost[i-1][k]+p1*abs(60.0*fs/j-60.0*fs/k)-tempo_candidates[i][j]:
                    cost[i][j]=cost[i-1][k]+p1*abs(60.0*fs/j-60.0*fs/k)-tempo_candidates[i][j]
                    m[i][j]=int(k)
                    
    tempo_period=zeros(T)
    tempo_period[T-1]=argmin(cost[T-1,1:])/float(fs)
    t=int(m[T-1,argmin(cost[T-1,1:])])
    i=T-2
    while(i>=0):
        tempo_period[i]=t/float(fs)
        t=int(m[i][t])
        i=i-1
    return tempo_period

def display_ACF(ACF, total_time, t_ACF_lag, GT=[], take_log=0, heading="", C='b'):
    plt.figure(figsize=(7, 4.5))
    if take_log == 0:
        plt.imshow(-ACF.transpose(), extent=[0, total_time, 0, t_ACF_lag], cmap=cm.Greys_r, origin='lower', aspect='auto')
    else:
        plt.imshow(-log(ACF.transpose()), extent=[0, total_time, 0, t_ACF_lag], cmap=cm.Greys_r, origin='lower', aspect='auto')
    plt.title(heading, fontsize=16, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.xticks(arange(0, total_time, 100))
    plt.ylabel('Lag (s)', fontsize=12, fontweight='bold')
    for i in range(0, len(GT)):
        plt.axvline(x=GT[i], ymin=0.0, ymax=1.0, linewidth=2, color=C, ls='--')
    plt.tight_layout()

def display_met_surf_tempo(time, total_time, metric_tempo_period, surface_tempo_period, GT=[], heading=""):
    #figure(figsize=(10, 5.5))
    m1, = plt.plot(time, 60/metric_tempo_period, marker='.', ms=1, ls='None', color='r')
    m2, = plt.plot(time, 60/surface_tempo_period, marker='.', ms=1, ls='None', color='k')
    plt.title(heading, fontsize=16, fontweight='bold')     
    plt.ylim([0, 1300])
    plt.xlabel('Time (s)', fontsize=15, fontweight='bold')
    plt.ylabel('Tempo (bpm)', fontsize=15, fontweight='bold')
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.legend((m1, m2), ('Metric Tempo','Surface Tempo'), scatterpoints=1, loc='upper left', ncol=1, fontsize=12)    
    for i in range( len(GT)):
        plt.axvline(x=GT[i], ymin=0.0, ymax=1.0, linewidth=3, color='b', ls='--')
    plt.xlim([0, total_time])
    plt.tight_layout()
    
def find_onsets(nov, Fs_nov, prominence=0.02):

    peaks, properties = sig.find_peaks(nov, prominence=0.02)
    T_coef = np.arange(nov.shape[0]) / Fs_nov
    peaks_sec = T_coef[peaks]

    return peaks, T_coef, peaks_sec


def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer
    
    

#%%
def compute_sm_dot(X,Y):
    """Computes similarty matrix from feature sequences using dot (inner) product
    Notebook: C4/C4S2_SSM.ipynb
    """
    S = np.dot(np.transpose(X), Y)
    return S


def compute_plot_spectrogram(x, Fs=22050, N=4096, H=2048, ylim=None,
                     figsize =(5, 2), title='', log=False):
    N, H = 1024, 512
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann',
                     center=True, pad_mode='constant')
    Y = np.abs(X)**2
    if log:
        Y_plot = np.log(1 + 100 * Y)
    else:
        Y_plot = Y
    FMP.libfmp.b.plot_matrix(Y_plot, Fs=Fs/H, Fs_F=N/Fs, title=title, figsize=figsize)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.show()
    return Y





def median_filter_horizontal(x, filter_len):
    """Apply median filter in horizontal direction
    Notebook: C8/C8S1_HPS.ipynb
    """
    return signal.medfilt(x, [1, filter_len])

def median_filter_vertical(x, filter_len):
    """Apply median filter in vertical direction
    Notebook: C8/C8S1_HPS.ipynb
    """
    return signal.medfilt(x, [filter_len, 1])

def plot_spectrogram_hp(Y_h, Y_p, Fs=22050, N=4096, H=2048, figsize =(10, 2),
                         ylim=None, clim=None, title_h='', title_p='', log=False):
    if log:
        Y_h_plot = np.log(1 + 100 * Y_h)
        Y_p_plot = np.log(1 + 100 * Y_p)
    else:
        Y_h_plot = Y_h
        Y_p_plot = Y_p
    plt.figure(figsize=figsize)
    ax = plt.subplot(1,2,1)
    FMP.libfmp.b.plot_matrix(Y_h_plot, Fs=Fs/H, Fs_F=N/Fs, ax=[ax], clim=clim,
                         title=title_h, figsize=figsize)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax = plt.subplot(1,2,2)
    FMP.libfmp.b.plot_matrix(Y_p_plot, Fs=Fs/H, Fs_F=N/Fs, ax=[ax], clim=clim,
                         title=title_p, figsize=figsize)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.show()




def compute_kernel_checkerboard_box(L):
    """Compute box-like checkerboard kernel [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L: Parameter specifying the kernel size 2*L+1

    Returns:
        kernel: Kernel matrix of size (2*L+1) x (2*L+1)
    """
    axis = np.arange(-L, L+1)
    kernel = np.outer(np.sign(axis), np.sign(axis))
    return kernel

#@jit(nopython=True)
def compute_kernel_checkerboard_gaussian(L, var=1, normalize=True):
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1]
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L: Parameter specifying the kernel size M=2*L+1
        var: Variance parameter determing the tapering (epsilon)

    Returns:
        kernel: Kernel matrix of size M x M
    """
    taper = np.sqrt(1/2) / (L * var)
    axis = np.arange(-L, L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel


def compute_novelty_ssm(S, kernel=None, L=10, var=0.5, exclude=False):
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S: SSM
        kernel: Checkerboard kernel (if kernel==None, it will be computed)
        L: Parameter specifying the kernel size M=2*L+1
        var: Variance parameter determing the tapering (epsilon)
        exclude: Sets the first L and last L values of novelty function to zero

    Returns:
        nov: Novelty function
    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2*L + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, L, mode='constant')

    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)
    if exclude:
        right = np.min([L, N])
        left = np.max([0, N-L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov


def generate_gaussian_output_vector(gt_ann, t_ACF, std=20, ln=41):
    ''' Output for LSTM Training'''
    peaks=gt_ann*2
    gauss_window = signal.windows.gaussian(ln, std)
    imp = signal.unit_impulse(len(t_ACF), peaks)
    filtered = signal.convolve(imp, gauss_window, mode='same')

    return filtered

    
