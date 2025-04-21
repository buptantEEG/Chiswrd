import os
import mne
from mne.time_frequency import tfr_multitaper, tfr_morlet
import numpy as np
from utilitys import *

TFR_method = "Morlet"
fmin = 0.5
fmax = 100
n_steps = 300
freqs = np.logspace(*np.log10([fmin, fmax]), num=n_steps)
n_cycles = freqs
average = True
return_itc = True
use_fft = False
decim = 1
n_jobs = 1
zero_mean = True
overwrite = True

picks = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4', 'FT7',\
'FT8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'CP3', 'CP4', 'TP7', 'TP8',\
'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2']

root_path  = "xxx"
save_path = 'xxx'
sub_files = os.listdir(root_path)
for sub in sub_files:
    ses_files = os.listdir(root_path+sub)
    for ses in ses_files:
        path = os.path.join(root_path, sub, ses)
        save_path_se = os.path.join(save_path, sub, ses)
        fif_file = path + '/eeg-epo.fif'
        print(fif_file)
        X_S = mne.read_epochs(fif_file, verbose='WARNING')
        Adquisition_eq = "standard_1020"
        montage = mne.channels.make_standard_montage(Adquisition_eq)
        X_S.set_montage(montage)

        if TFR_method == "Multitaper":
            power, itc = tfr_multitaper(X_S, freqs=freqs, n_cycles=n_cycles,
                                        use_fft=use_fft, return_itc=return_itc,
                                        decim=decim, n_jobs=n_jobs,
                                        average=average, picks=picks)

            ensure_dir(save_path_se)
            file_name = save_path_se + "/power-tfr.h5"
            power.save(fname=file_name, overwrite=overwrite)

            file_name = save_path_se + "/itc-tfr.h5" 
            itc.save(fname=file_name, overwrite=overwrite)

        elif TFR_method == "Morlet":
            power, itc = tfr_morlet(X_S, freqs=freqs, n_cycles=n_cycles,
                                    use_fft=use_fft, return_itc=return_itc,
                                    decim=decim, n_jobs=n_jobs,
                                    average=average, zero_mean=zero_mean,
                                    picks=picks)
            ensure_dir(save_path_se)
            file_name = save_path_se + "/power-tfr.h5"
            print(file_name)
            power.save(fname=file_name, overwrite=overwrite)

            file_name = save_path_se + "/itc-tfr.h5"  
            itc.save(fname=file_name, overwrite=overwrite)
            print(file_name)
        else:
            print("Invalid TFR_rep")
