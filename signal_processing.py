try:
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import inv as inv
    from scipy.fftpack import fft, ifft
    
except ImportError:
    print('Make sure you have all the libraries properly installed')
    exit()



# Create detrending matrix
def create_detrending_matrix(signal_length=0, lambd=50):
    if (signal_length<=0):
        print('In function create_detrending_matrix: signal length is <=0')
        exit()

    T = signal_length
    I = sp.eye(T)
    D2 = sp.spdiags( ([1,-2,1]*np.ones((1,T-2),dtype=np.int).T).T, (range(0,3)),T-2,T)

    return (I-inv(I+lambd*lambd*D2.T*D2))



# Detrending
def detrend(signals, D):
    for index_signals in range(signals.shape[0]):
        signals[index_signals] = signals[index_signals] - np.mean(signals[index_signals])
        signals[index_signals] = D * signals[index_signals]

    return signals



# Compute signal to noise ratio from detrended PPG signals
def compute_SNR(signals, lim_skin_detect=5, framerate=30):
    SNR = np.zeros((lim_skin_detect))
    
    # Compute SNR for each mask
    for index_masks in range(lim_skin_detect):
        xdft = fft(signals[index_masks])
        xdft = xdft[1:int(signals.shape[1]/2+1)]
        PSD = np.abs(xdft)**2
        PSD = PSD[int(0.7*signals.shape[1]/framerate):int(4*signals.shape[1]/framerate)]

        SNR[index_masks] = np.max(PSD) / (np.sum(PSD) - np.max(PSD))

    ind = np.argsort(-SNR)

    return SNR, ind