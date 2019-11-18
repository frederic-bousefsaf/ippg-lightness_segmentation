try:
    import os
    import numpy as np
    import cv2

except ImportError:
    print('Make sure you have all the libraries properly installed')
    exit()


# Load images and compute PPG signals using segmented masks (based on lightness distribution)
def load_process_images_from_path(path, detector, predictor, lim_skin_detect=5, nb_frames_to_average=5, display_marks=True):
    
    ## VAR 
    last_landmarks = []
    circle_roi = []
    rectangle_roi = []

    ## LOAD IMAGES
    list_dir = os.listdir(path)

    # Load an image to get the dimensions
    temp = cv2.imread(path + '/' + list_dir[0], cv2.IMREAD_ANYCOLOR)
    IMAGE_WIDTH = temp.shape[1]
    IMAGE_HEIGHT = temp.shape[0]
    IMAGE_CHANNELS = temp.shape[2]
    IMAGE_TOTAL_NUMBER = len(list_dir)

    img = np.zeros((IMAGE_TOTAL_NUMBER, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    signals = np.zeros( (lim_skin_detect, IMAGE_TOTAL_NUMBER-nb_frames_to_average+1) )

    for index_images in range(IMAGE_TOTAL_NUMBER):
        img = cv2.imread(path + '/' + list_dir[index_images], cv2.IMREAD_ANYCOLOR)
        img_d = img.copy()
        cv2.GaussianBlur(img, (5, 5), 0, img, 0, 4)


        # DETECT FACE AND COMPUTE LANDMARKS USING DLIB
        landmarks = landmarks_detection(img, detector, predictor)
        last_landmarks.append(landmarks)


        ## AVERAGE THE LAST N SHAPES (POINTS STABILIZATION)
        if (index_images >= nb_frames_to_average-1):
            averaged_landmarks, img_d = landmarks_averaging(landmarks, last_landmarks, nb_frames_to_average, display_marks, img_d)
            last_landmarks.pop(0)
           
            ## PREPARE ELLIPSE POINTS
            bounding_pts, img_d = ellipse_points_from_landmarks(averaged_landmarks, display_marks, img_d)


            ## COMPUTE ELLIPTICAL MASK AND ROI
            circle_roi, rectangle_roi, img_roi, img_roi_lightness, img_roi_u, mask_roi = compute_mask_and_roi_from_ellipse_points(img, bounding_pts, circle_roi, rectangle_roi)


            ## COMPUTE HISTOGRAM
            hist_limits = compute_histogram(img_roi_lightness, mask_roi, lim_skin_detect)


            # COMPUTE MASKS AND FORM SIGNAL
            signals[:, index_images-nb_frames_to_average+1] = spatial_averaging(img_roi_lightness, img_roi_u, mask_roi, hist_limits, lim_skin_detect)


        print("image " + str(index_images+1) + " over " + str(IMAGE_TOTAL_NUMBER) + " has been processed")

        if (display_marks):
            cv2.imshow('src', img_d)
            cv2.waitKey(1)

    return circle_roi, rectangle_roi, signals



# Load images and compute PPG signals using combined masks
def load_process_images_with_combined_masks_from_path(path, circle_roi, rectangle_roi, SNR, sorted_index, lim_skin_detect=5, nb_frames_to_average=5, display_marks=True):
    
    # This function is pretty similar to "load_process_images_from_path"
    list_dir = os.listdir(path)

    signals = np.zeros( (lim_skin_detect-1, len(list_dir)-nb_frames_to_average+1) )

    for index_images in range(nb_frames_to_average-1, len(list_dir)):
        img = cv2.imread(path + '/' + list_dir[index_images], cv2.IMREAD_ANYCOLOR)
        cv2.GaussianBlur(img, (5, 5), 0, img, 0, 4)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.ellipse(mask, circle_roi[index_images-(nb_frames_to_average-1)], (255), -1, 1)

        img_roi = img[rectangle_roi[index_images-(nb_frames_to_average-1)][1]:rectangle_roi[index_images-(nb_frames_to_average-1)][1]+rectangle_roi[index_images-(nb_frames_to_average-1)][3], rectangle_roi[index_images-(nb_frames_to_average-1)][0]:rectangle_roi[index_images-(nb_frames_to_average-1)][0]+rectangle_roi[index_images-(nb_frames_to_average-1)][2], :]
        mask_roi = mask[rectangle_roi[index_images-(nb_frames_to_average-1)][1]:rectangle_roi[index_images-(nb_frames_to_average-1)][1]+rectangle_roi[index_images-(nb_frames_to_average-1)][3], rectangle_roi[index_images-(nb_frames_to_average-1)][0]:rectangle_roi[index_images-(nb_frames_to_average-1)][0]+rectangle_roi[index_images-(nb_frames_to_average-1)][2]]

        img_roi_Luv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2Luv)
        img_roi_lightness = img_roi_Luv[:,:,0]
        img_roi_u = img_roi_Luv[:,:,1]

        hist_limits = compute_histogram(img_roi_lightness, mask_roi, lim_skin_detect)
        
        signals[:, index_images-nb_frames_to_average+1] = spatial_averaging_from_combined_masks(img_roi_lightness, img_roi_u, mask_roi, hist_limits, sorted_index, lim_skin_detect)

    return signals



# Detect facial landmarks
def landmarks_detection(img, detector, predictor):
    
    faces = detector(img, 1)

    if (faces):
        face = faces[0]
        
    shape = predictor(img, face)

    landmarks = []
    
    for index_landmarks in range(68):
        landmarks.append(shape.part(index_landmarks))
    
    return landmarks



# Average landmarks
def landmarks_averaging(landmarks, last_landmarks, nb_frames_to_average, display_marks, img_d):
    
    averaged_landmarks = np.zeros((68, 2), dtype=int)

    for index_landmarks in range(68):
        for index_frames_to_average in range(0, nb_frames_to_average):
            averaged_landmarks[index_landmarks][0] = averaged_landmarks[index_landmarks][0] + last_landmarks[index_frames_to_average][index_landmarks].x
            averaged_landmarks[index_landmarks][1] = averaged_landmarks[index_landmarks][1] + last_landmarks[index_frames_to_average][index_landmarks].y

        averaged_landmarks[index_landmarks] = np.round(averaged_landmarks[index_landmarks] / nb_frames_to_average)

        if (display_marks):
                cv2.circle(img_d, (averaged_landmarks[index_landmarks][0], averaged_landmarks[index_landmarks][1]), 1, (0,0,255), thickness=2)    

    return averaged_landmarks, img_d



# Ellipse from landmarks
def ellipse_points_from_landmarks(landmarks, display_marks, img_d):
    
    bounding_pts = np.zeros((9, 2), dtype=int)
        
    bounding_pts[0] = np.round(landmarks[27] - (landmarks[33] - landmarks[27]) * 1.25)
    bounding_pts[1] = landmarks[16]
    bounding_pts[2] = landmarks[0]
    bounding_pts[3] = landmarks[33]

    bounding_pts[4][0] = bounding_pts[0][0] - (bounding_pts[0][0] - bounding_pts[3][0]) / 2
    bounding_pts[4][1] = bounding_pts[2][1] - (bounding_pts[2][1] - bounding_pts[1][1]) / 2

    bounding_pts[5] = bounding_pts[0] + bounding_pts[1] - bounding_pts[4]
    bounding_pts[6] = bounding_pts[3] + bounding_pts[1] - bounding_pts[4]
    bounding_pts[7] = bounding_pts[2] + bounding_pts[3] - bounding_pts[4]
    bounding_pts[8] = bounding_pts[2] + bounding_pts[0] - bounding_pts[4]

    if (display_marks):
        for index_pts in range(9):
            cv2.circle(img_d, (bounding_pts[index_pts][0], bounding_pts[index_pts][1]), 1, (0,255,0), thickness=2)

    return bounding_pts, img_d



# Mask and ROI from ellipse points
def compute_mask_and_roi_from_ellipse_points(img, bounding_pts, circle_roi, rectangle_roi):
    
    circle_roi.append(cv2.minAreaRect(bounding_pts))
    rectangle_roi.append(cv2.boundingRect(bounding_pts))

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.ellipse(mask, circle_roi[-1], (255), -1, 1)

    img_roi = img[rectangle_roi[-1][1]:rectangle_roi[-1][1]+rectangle_roi[-1][3], 
                  rectangle_roi[-1][0]:rectangle_roi[-1][0]+rectangle_roi[-1][2], :]

    mask_roi = mask[rectangle_roi[-1][1]:rectangle_roi[-1][1]+rectangle_roi[-1][3], 
                    rectangle_roi[-1][0]:rectangle_roi[-1][0]+rectangle_roi[-1][2]]

    img_roi_Luv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2Luv)
    img_roi_lightness = img_roi_Luv[:,:,0]
    img_roi_u = img_roi_Luv[:,:,1]

    return circle_roi, rectangle_roi, img_roi, img_roi_lightness, img_roi_u, mask_roi



# Compute histogram ranges
def compute_histogram(img_roi_lightness, mask_roi, lim_skin_detect):
    
    hist = cv2.calcHist([img_roi_lightness], [0], mask_roi, [256], [0, 256])

    count = np.sum(mask_roi) / (255 * lim_skin_detect)
    total_number_masks = 1
    cumsum = 0

    hist_limits = np.zeros((lim_skin_detect+1), dtype=np.uint8)

    for index_bins in range(len(hist)):
        cumsum = cumsum + hist[index_bins][0]

        if (cumsum > count):
            cumsum = 0
            hist_limits[total_number_masks] = index_bins
            total_number_masks = total_number_masks + 1
    hist_limits[total_number_masks] = 255

    return hist_limits



# Compute signals from a frame and the corresponding masks
def spatial_averaging(img_roi_lightness, img_roi_u, mask_roi, hist_limits, lim_skin_detect):
    
    signals = np.zeros(lim_skin_detect)

    for index_masks in range(lim_skin_detect):
        img_skin_mask = ((img_roi_lightness >= hist_limits[index_masks]) & (img_roi_lightness < hist_limits[index_masks+1]))*mask_roi
        signals[index_masks] = cv2.mean(img_roi_u, img_skin_mask)[0]

    return signals


# Compute signal from a frame and combined masks
def spatial_averaging_from_combined_masks(img_roi_lightness, img_roi_u, mask_roi, hist_limits, sorted_index, lim_skin_detect):
    
    signals = np.zeros(lim_skin_detect-1)

    img_skin_mask = ((img_roi_lightness >= hist_limits[sorted_index[0]]) & (img_roi_lightness < hist_limits[sorted_index[0]+1]))*mask_roi

    for index_masks in range(lim_skin_detect-1):
        img_skin_mask = img_skin_mask + ((img_roi_lightness >= hist_limits[sorted_index[index_masks+1]]) & (img_roi_lightness < hist_limits[sorted_index[index_masks+1]+1]))*mask_roi
        signals[index_masks] = cv2.mean(img_roi_u, img_skin_mask)[0]

    return signals


