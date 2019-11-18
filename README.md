# Automatic Selection of Webcam Photoplethysmographic Pixels Based on Lightness Criteria

Remote pulse rate measurement from facial video has gained particular attention over the last few years. Researches exhibit significant advancements and demonstrate that common video cameras correspond to reliable devices that can be employed to measure a large set of biomedical parameters without any contact with the subject.

This repository contains the source codes related to a method based on a prior selection of pixels of interest using a custom segmentation that used the face lightness distribution to define different sub-regions. The most relevant sub-regions are automatically selected and combined by evaluating their respective signal to noise ratio.


## Reference
If you find this code useful or use it in an academic or research project, please cite it as: 

Frédéric Bousefsaf, Alain Pruski, Choubeila Maaoui, **Automatic selection of webcam photoplethysmographic pixels based on lightness criteria**, *Journal of Medical and Biological Engineering*, vol. 37, n° 3, pp. 374–385 (2017). [Link](https://www.researchgate.net/publication/308200884_Automatic_Selection_of_Webcam_Photoplethysmographic_Pixels_Based_on_Lightness_Criteria)

You can also visit my [website](https://sites.google.com/view/frederic-bousefsaf) for additional information.

## Scientific description
Please refer to the original publication to get all the details. The method automatically selects different regions of pixels based on the lightness distribution of the face. The signal to noise ratio of each region is computed using a standard power spectral density analysis. The most relevant regions are then automatically selected and combined by evaluating their respective signal to noise ratio. To avoid artifacts generated during lips movements, only the upper part of the face was selected as first ROI. 

![Alt text](illustrations/method.png?raw=true "Method")

*Overview of the method.*


## Requirements
The codes were developped and tested with Python 3.5/3.6. The Computer Vision System Toolbox is required for face detection and tracking.


## Usage
Function inputs: 
- `file`: source folder path (.png images) or video path/filename.
- `mode`: 'video' or 'folder'. If 'folder' is specified, images must follow a %04d template that starts from 0, i.e. '0000.png', '0001.png'...
- `display`: 0 = no display, 1 = display signals only, 2 = display signals and face tracking.


Function outputs: 
- `iPPG_time30`, `iPPG_signal30filt`: iPPG signal and time vectors (u* channel filtered using its CWT representation).
- `iPR_time`, `iPR`: instantaneous (beat-to-beat) pulse rate.
- `iBR_time`, `iBR`: instantaneous (beat-to-beat) breathing rate.

Below is a typical usage example. A test sample is available  [here](https://drive.google.com/open?id=17l_MJVqw4F9cQpcJ-_wFmFNr3bdZNtw9) (sample_front.zip). The folder contains the time vector along with uncompressed images. 

`[iPPG_time30, iPPG_signal30filt, iPR_time, iPR, iBR_time, iBR] = ippg_luv_skindetect_cwt('C:\sample_front', 'folder', 1);`

![Alt text](illustrations/results.png?raw=true "Results computed from the sample data")

*Results computed from the sample data.*
