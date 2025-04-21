import os

def ensure_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def info_sub(sub):
    sesseion = None
    DS_rate = None
    sfreq = None
    if sub == 1:
        sesseion = [2]
        sfreq = 1000
        DS_rate = 4
    elif sub == 2:
        sesseion = [1]
        sfreq = 1000
        DS_rate = 4
    elif sub == 3:
        sesseion = [1, 2, 3, 4, 5]
        sfreq = 1000
        DS_rate = 4
    elif sub == 4:
        sesseion = [1, 2, 3, 4, 5]
        sfreq = 1000
        DS_rate = 4
    elif sub == 5:
        sesseion = [1, 2, 3, 4, 5]
        sfreq = 1000
        DS_rate = 4
    elif sub == 6:
        sesseion = [1, 2, 3, 4]
        sfreq = 1000
        DS_rate = 4
    elif sub == 7:
        sesseion = [1, 2, 4, 5]
        sfreq = 500
        DS_rate = 2
    elif sub == 8:
        sesseion = [1, 3, 4]
        sfreq = 500
        DS_rate = 2
    elif sub == 9:
        sesseion = [1, 2, 3, 4, 5]
        sfreq = 500
        DS_rate = 2
    elif sub == 10:
        sesseion = [1, 2, 3, 4, 5]
        sfreq = 500
        DS_rate = 2
    elif sub == 11:
        sesseion = [1, 2, 3, 4, 5]
        sfreq = 500
        DS_rate = 2
    elif sub == 12:
        sesseion = [1, 2, 3, 4, 5]
        sfreq = 500
        DS_rate = 2
    return sesseion, sfreq, DS_rate