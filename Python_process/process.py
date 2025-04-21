import os
import mne
import pickle
import sys

import numpy as np
sys.path.append('xxx')
from event_gener import *
from utilitys import *

# #################### Filtering
# Cut-off frequencies
Low_cut = 0.5
High_cut = 100

# Notch filter in 50Hz
Notch_bool = True

# #################### ICA
# If False, ICA is not applyed
ICA_bool = True
ICA_Components = None
ica_random_state = 23
ica_method = 'infomax'
fit_params = dict(extended=True)

event_id_all = {
        "fixation": 13,
        "close_eyes": 15,
        "concentrate_data": 21,
        "action_data": 22,
        "rest_data": 23,
        "我": 30,
        "你": 31,
        "他": 32,
        "上": 33,
        "下": 34,
        "左": 35,
        "右": 36,
        "是": 37,
        "水": 38,
        "饭": 39}
# event_dic = {"我": 30,
#         "你": 31,
#         "他": 32,
#         "上": 33,
#         "下": 34,
#         "左": 35,
#         "右": 36,
#         "是": 37,
#         "水": 38,
#         "饭": 39}
event_dic = {"30": 30,
        "31": 31,
        "32": 32,
        "33": 33,
        "34": 34,
        "35": 35,
        "36": 36,
        "37": 37,
        "38": 38,
        "39": 39}
eye_open_id = {"13": 13}
eyc_closed_id = {"15": 15}

ref_channels = ['CPz']
eye_channels = ['HEOL', 'VEOR']
ch_names = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4', 'FT7',\
'FT8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'CP3', 'CP4', 'TP7', 'TP8',\
'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2', 'CPz', 'HEOL', 'VEOR']
ch_types = ['eeg', 'eeg', 'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',\
            'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg',\
            'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eog','eog']

root_path = 'xxx'
for sub in range(1, 13):
        sesseions, sfreq, DS_rate = info_sub(sub)
        for ses in sesseions:
                path = root_path + f'sub-{sub}/ses-{ses}/'
                save_path = root_path + f'derivatives/sub-{sub}/ses-{ses}/'
                print(f'sub and ses are {sub} and {ses}')
                event_np, all_data = event_generate(path)
                # mon = mne.channels.make_standard_montage("biosemi32")
                # print(mon.ch_names)

                info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
                rawdata = mne.io.RawArray(all_data, info)
                rawdata.evnet = event_np

                # # # Referencing
                # rawdata.set_eeg_reference([])

                if Notch_bool:
                        rawdata = mne.io.Raw.notch_filter(rawdata, freqs=50)
                rawdata.filter(Low_cut, High_cut)

                #EOG
                picks_eog = mne.pick_types(rawdata.info, eeg=False, stim=False, include=['HEOL', 'VEOR'])
                epochsEOG = mne.Epochs(rawdata, event_np, event_id=event_dic, tmin=-2,
                                        tmax=8, picks=picks_eog, preload=True,
                                        detrend=0, decim=DS_rate, baseline=None)
                ensure_dir(save_path)
                file_name = save_path +'eog-epo.fif'  
                epochsEOG.save(file_name, fmt='double', split_size='2GB', overwrite=True)
                del epochsEOG

                #eye_open restting state
                picks_eeg_rest = mne.pick_types(rawdata.info, eeg=True, exclude=['CPz', 'HEOL', 'VEOR'], stim=False)
                epochs_eye_open = mne.Epochs(rawdata, event_np, event_id=eye_open_id, tmin=0,
                                tmax=60, picks=picks_eeg_rest, preload=True, detrend=0, decim=DS_rate, baseline=None)
                file_name = save_path + 'eye_open-epo.fif'
                epochs_eye_open.save(file_name, fmt='double', split_size='2GB', overwrite=True)

                #eye_closed restting state
                epochs_eye_closed = mne.Epochs(rawdata, event_np, event_id=eyc_closed_id, tmin=0,
                                tmax=60, picks=picks_eeg_rest, preload=True, detrend=0, decim=DS_rate, baseline=None)
                file_name = save_path + 'eye_closed-epo.fif'
                epochs_eye_closed.save(file_name, fmt='double', split_size='2GB', overwrite=True)

                #EEG
                picks_eeg = mne.pick_types(rawdata.info, eeg=True, exclude=['CPz', 'HEOL', 'VEOR'], stim=False)
                epochsEEG = mne.Epochs(rawdata, event_np, event_id=event_dic, tmin=4,
                                        tmax=8, picks=picks_eeg, preload=True,
                                        detrend=0, decim=DS_rate, baseline=None)
                # ICA preprosessing
                if ICA_bool:
                        picks_vir = mne.pick_types(rawdata.info, eeg=True, include=['HEOL', 'VEOR'], stim=False)
                        epochsEEG_full = mne.Epochs(rawdata, event_np, event_id=event_dic,
                                                tmin=4, tmax=8,
                                                picks=picks_vir, preload=True,
                                                detrend=0, decim=DS_rate,
                                                baseline=None)
                        ica = mne.preprocessing.ICA(n_components=ICA_Components,
                                                random_state=ica_random_state,
                                                method=ica_method,
                                                fit_params=fit_params)
                        ica.fit(epochsEEG)
                        ica.exclude = []
                        exg_inds_HEOL, scores_ica = ica.find_bads_eog(epochsEEG_full, ch_name='HEOL')
                        ica.exclude.extend(exg_inds_HEOL)
                        exg_inds_VEOR, scores_ica = ica.find_bads_eog(epochsEEG_full, ch_name='VEOR')
                        ica.exclude.extend(exg_inds_VEOR)

                        print("Appling ICA")
                        ica.apply(epochsEEG)
                ensure_dir(save_path)
                file_name = save_path +'eeg-epo.fif' 
                epochsEEG.save(file_name, fmt='double', split_size='2GB', overwrite=True)
                print(f'sub-{sub}/ses-{ses} is done')

