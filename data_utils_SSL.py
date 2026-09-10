import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import soundfile as sf
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        key = parts[1]
        
        if is_eval:
            file_list.append(key)
        else:
            # Robustly find 'bonafide' or 'spoof' from the end of the line
            label_val = None
            for token in reversed(parts):
                clean_token = token.lower()
                if clean_token in ['bonafide', 'spoof']:
                    label_val = 1 if clean_token == 'bonafide' else 0
                    break
            
            if label_val is not None:
                file_list.append(key)
                d_meta[key] = label_val

    if is_eval:
        return file_list
    return d_meta, file_list

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def load_audio(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32)

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo, aug=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = 64600 
        self.aug = aug

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        audio_path = os.path.join(self.base_dir, 'flac', f'{utt_id}.flac')
        
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.base_dir, 'wav', f'{utt_id}.wav')

        X = load_audio(audio_path)
        Y = process_Rawboost_feature(X, 16000, self.args, self.algo)
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        
        return x_inp, target
                     
class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600 

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        audio_path = os.path.join(self.base_dir, 'flac', f'{utt_id}.flac')
        
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.base_dir, 'wav', f'{utt_id}.wav')

        X = load_audio(audio_path)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id  

#--------------RawBoost data augmentation algorithms---------------------------##
def process_Rawboost_feature(feature, sr, args, algo):
    if algo == 1:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 3:
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 4:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)                         
        feature = ISD_additive_noise(feature, args.P, args.g_sd)  
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)                 
    elif algo == 5:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)                         
        feature = ISD_additive_noise(feature, args.P, args.g_sd)                
    elif algo == 6:  
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)                         
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr) 
    elif algo == 7: 
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr) 
    elif algo == 8:
        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)                         
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)
    else:
        feature = feature
    
    return feature
