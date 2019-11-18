try:
    import tkinter as tk
    from tkinter import filedialog
    import dlib
    import numpy as np
    import matplotlib.pyplot as plt

    import signal_processing
    import image_processing
except ImportError:
    print('Make sure you have all the libraries properly installed')
    exit()


# VARIABLES AND CONSTANTS
FRAMERATE = 30
NB_FRAMES_TO_AVERAGE = 5
LIM_SKIN_DETECT = 5
DISPLAY_MARKS = True

landmark_path = "shape_predictor_68_face_landmarks.dat" # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


# ASK USER FOR PATH USING A GUI
root = tk.Tk()
root.withdraw()
DIR_DATA = filedialog.askdirectory()
if(DIR_DATA==''):
    print('No folder has been selected')
    exit()

root.destroy()
del root


# INIT DLIB FACE DETECTOR AND FACIAL SHAPE PREDICTOR
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_path)
except ImportError:
    print('Error during dlib model loading (check if the file shape_predictor_68_face_landmarks is in the source folder)')
    exit()


## LOAD AND PROCESS IMAGES (FACE DETECTION - LANDMARKS EXTRACTION - LIGHTNESS SEGMENTATION - SIGNAL FORMATION)
circle_roi, rectangle_roi, signals = image_processing.load_process_images_from_path(DIR_DATA, detector, predictor, LIM_SKIN_DETECT, NB_FRAMES_TO_AVERAGE, DISPLAY_MARKS)


## DETREND SIGNALS
D = signal_processing.create_detrending_matrix(signals.shape[1])
signals = signal_processing.detrend(signals, D)


## COMPUTE SIGNAL TO NOISE RATIO
SNR, sorted_index = signal_processing.compute_SNR(signals, LIM_SKIN_DETECT, FRAMERATE)


## FORM COMBINED MASKS AND COMPUTE NEW SIGNALS
signals_combined_masks = image_processing.load_process_images_with_combined_masks_from_path(DIR_DATA, circle_roi, rectangle_roi, SNR, sorted_index, LIM_SKIN_DETECT, NB_FRAMES_TO_AVERAGE, DISPLAY_MARKS)


## DETREND SIGNALS AND COMPUTE SIGNAL TO NOISE RATIO
signals_combined_masks = signal_processing.detrend(signals_combined_masks, D)
SNR_combined_masks, sorted_index = signal_processing.compute_SNR(signals_combined_masks, LIM_SKIN_DETECT-1, FRAMERATE)

SNR = np.concatenate( (SNR, SNR_combined_masks) )
signals = np.concatenate( (signals, signals_combined_masks) )


## EXTRACT AND DISPLAY BEST PPG SIGNAL
sorted_index = np.argsort(-SNR)
signal = signals[sorted_index[0]]

plt.plot(signal)
plt.title('PPG signal with best SNR')
plt.xlabel('Frame')
plt.ylabel('n.u.')
plt.show()
