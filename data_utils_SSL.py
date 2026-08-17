"""
data_utils_SSL.py — adapted for MSS-Urdu dataset
Protocol format: SPEAKER_ID  UTT_ID  -  SYSTEM_ID  bonafide/spoof
                 col[0]      col[1]       col[3]    col[4]
"""

import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import soundfile as sf
from RawBoost import (ISD_additive_noise, LnL_convolutive_noise,
                      SSI_additive_noise, normWav)

SAMPLE_RATE = 16_000
TRIM_LENGTH  = 64_600


def pad_or_trim(x, length=TRIM_LENGTH):
    if len(x) >= length:
        return x[:length]
    return np.tile(x, (length // len(x)) + 1)[:length]


def load_audio(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def _find_audio(base_dir, utt_id):
    for ext in ('.flac', '.wav'):
        p = os.path.join(base_dir, utt_id + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Audio not found: {utt_id} in {base_dir}")


def _parse_line(line):
    # SPEAKER_ID  UTT_ID  -  SYSTEM_ID  bonafide/spoof
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None
    utt_id = parts[1]
    label  = 1 if parts[4].lower() == 'bonafide' else 0
    return utt_id, label


def genSpoof_list(dir_meta, is_train, is_eval):
    label_dict, file_list = {}, []
    with open(dir_meta) as f:
        for line in f:
            utt_id, label = _parse_line(line)
            if utt_id is None:
                continue
            file_list.append(utt_id)
            label_dict[utt_id] = label
    if is_eval:
        return file_list
    return label_dict, file_list


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels   = labels
        self.base_dir = base_dir
        self.algo     = algo
        self.args     = args
        self.cut      = TRIM_LENGTH
        self.sr       = SAMPLE_RATE

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        label  = self.labels[utt_id]
        x = load_audio(_find_audio(self.base_dir, utt_id))
        x = pad_or_trim(x, self.cut)
        x = process_Rawboost_feature(x, self.sr, self.args, self.algo)
        return Tensor(x), label


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut      = TRIM_LENGTH

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        x = load_audio(_find_audio(self.base_dir, utt_id))
        x = pad_or_trim(x, self.cut)
        return Tensor(x), utt_id


def process_Rawboost_feature(feature, sr, args, algo):
    if algo == 0:
        return feature
    elif algo == 1:
        return LnL_convolutive_noise(feature, args.N_f, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
    elif algo == 2:
        return ISD_additive_noise(feature, args.P, args.g_sd)
    elif algo == 3:
        return SSI_additive_noise(feature, args.SNRmin, args.SNRmax,
            args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 4:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        return SSI_additive_noise(feature, args.SNRmin, args.SNRmax,
            args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    elif algo == 5:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        return normWav(feature, 0)
    elif algo == 6:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax,
            args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
        return normWav(feature, 0)
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax,
            args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
        return normWav(feature, 0)
    elif algo == 8:
        f1 = LnL_convolutive_noise(feature, args.N_f, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        f2 = ISD_additive_noise(feature, args.P, args.g_sd)
        return normWav((f1 + f2) / 2, 0)
    else:
        raise ValueError(f"Unknown algo: {algo}")
