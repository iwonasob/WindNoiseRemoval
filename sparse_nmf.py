from nmf import *
from pre_processing import *
from istft import *

def sparse_nmf(mixture, fs_mix ):
    ## training parameters
    N_s=64
    N_n=64
    lam_n_train=0
    lam_n=0
    lam_s=0
    file_train = os.path.join('data', 'train', 'wind_train_1min.wav')
    [fs_train, s_train] = read(file_train)

    [magnitude_wind_norm, phase, n_fft, hop_size, gamma]=prepare_data(s_train)

    nmf_model=NMF(N_n, norm_D=1,  iterations=100)
    [D_n,H_n,error]=nmf_model.process(magnitude_wind_norm, lam_n_train)

    ## nmf mixture decomposition
    [magnitude_mix_norm, phase, n_fft, hop_size, gamma] = prepare_data(s_train)
    nmf_model=NMF(N_s+N_n, update_func="kl_mix", norm_D=1,  iterations=500)
    [D,H,error]=nmf_model.process_mix(magnitude_mix_norm, lam_s, lam_n, D_n, N_s, N_n)

    D_s = D[:, -N_s:]
    H_s = H[-N_s:, :]

    ## denoise the signal
    X_s=np.dot(D_s,H_s)

    istft=ISTFT( window=None, fft_size=n_fft, hop_size=hop_size, sample_rate=fs_train)
    y_s=istft.process(np.sqrt(X_s)*phase)
    y_s=y_s/np.max(y_s)

    return y_s