# Automatic Selection of Webcam Photoplethysmographic Pixels Based on Lightness Criteria

Remote pulse rate measurement from facial video has gained particular attention over the last few years. Researches exhibit significant advancements and demonstrate that common video cameras correspond to reliable devices that can be employed to measure a large set of biomedical parameters without any contact with the subject.

This repository contains the source codes related to a method based on a prior selection of pixels of interest using a custom segmentation that used the face lightness distribution to define different sub-regions. The most relevant sub-regions are automatically selected and combined by evaluating their respective signal to noise ratio.


## Reference
If you find this code useful or use it in an academic or research project, please cite it as: 

Frédéric Bousefsaf, Choubeila Maaoui, Alain Pruski, **Automatic selection of webcam photoplethysmographic pixels based on lightness criteria**, *Journal of Medical and Biological Engineering*, vol. 37, n° 3, pp. 374–385 (2017). [Link](https://www.researchgate.net/publication/308200884_Automatic_Selection_of_Webcam_Photoplethysmographic_Pixels_Based_on_Lightness_Criteria)

You can also visit my [website](https://sites.google.com/view/frederic-bousefsaf) for additional information.

## Scientific description
Please refer to the original publication to get all the details. The method automatically selects different regions of pixels based on the lightness distribution of the face. The signal to noise ratio of each region is computed using a standard power spectral density analysis. The most relevant regions are then automatically selected and combined by evaluating their respective signal to noise ratio. To avoid artifacts generated during lips movements, only the upper part of the face was selected as first ROI. 

![Alt text](illustrations/method.png?raw=true "Method")

*Overview of the method.*


## Requirements
The codes were tested with Python 3.5/3.6 and Tensorflow + Keras frameworks.

Different packages must be installed to properly run the codes : 
- `pip install numpy`
- `pip install scipy`
- `pip install cmake`
- `pip install dlib`
- pip install opencv-python`
- `pip install matplotlib`


## Usage
A test sample is available  [here](https://drive.google.com/open?id=17l_MJVqw4F9cQpcJ-_wFmFNr3bdZNtw9) (sample_front.zip). The folder contains the time vector along with uncompressed images. Please remove the file `time.txt` from the folder before testing. Upon execution of `ippg_lightness_segmentation.py`. The program first displays a GUI. The folder containing only the raw images can be selected. PPG signals with best signal-to-noise ratio is automatically displayed at the end of the procedure (see the example below).

![Alt text](illustrations/results.png?raw=true "Results computed from the sample data")

*Results computed from the sample data.*
