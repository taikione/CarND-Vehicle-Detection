"""
Utilities to detect vehicles
parameters
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
ORIENT = 9
"""
import cv2
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):

    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys',
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)

        return features, hog_image

    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys', transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)

        return features


def bin_spatial(img, size=(32, 32)):

    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()

    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, orient, pix_per_cell, cell_per_block):

    # Create a list to append feature vectors
    features = []

    # Iterate through the list of images
    for file in imgs:

        # use png image
        image = mpimg.imread(file)
        #image = (image*255).astype(np.uint8) # 0-1 float32 to 0-255 uint8
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        #feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features = []

        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                 orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))

        hog_features = np.hstack((hog_features))

        # Get color features
        spatial_features = bin_spatial(feature_image)
        hist_features = color_hist(feature_image)

        # Scale features and make a prediction
        stacked = np.hstack((spatial_features, hist_features, hog_features))

        features.append(stacked)

    # Return list of feature vectors
    return features


def window_search(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                  cell_per_block, spatial_size=(32, 32), hist_bins=32):

    car_boxes = []
    # jpg image
    draw_img = np.copy(img)

    img_tosearch = img[ystart:ystop, :, :]
    img_tosearch = img_tosearch.astype(np.float32)/255.0

    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    #ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64

    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            stacked = np.hstack((spatial_features, hist_features, hog_features))

            test_features = X_scaler.transform(stacked.reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                car_boxes.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return car_boxes
    # return draw_img

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels, svc=None, vis_prob=False):

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):

        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

        if vis_prob == True:
            target = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0], :]
            prob = predict_window(target, svc)
            message = "{:.3f}".format(prob[0])
            # Draw probability of sub image
            cv2.putText(img, message, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)

    # Return the image
    return img

def predict_window(img, svc):
    img255 = img.astype(np.float32)/255.0
    img255 = cv2.resize(img255, (64, 64))
    feature_image = cv2.cvtColor(img255, cv2.COLOR_RGB2YCrCb)

    hog_features = []

    for channel in range(feature_image.shape[2]):
        hog_features.append(get_hog_features(feature_image[:,:,channel],
                                             orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
    hog_features = np.hstack((hog_features))

    spatial_features = bin_spatial(feature_image)
    hist_features = color_hist(feature_image)

    stacked = np.hstack((spatial_features, hist_features, hog_features))

    input_features = X_scaler.transform(stacked.reshape(1, -1))
    prob = svc.predict_proba(input_features)[0]

    return prob


def conpare_prev_frame(prev_bboxes, current_bboxes):
    """
    Remove bounding boxes that did not overlap with previous frame bounding boxes.
    each bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
    """

    overlap_bboxes = []
    for current_bbox in current_bboxes:

        overlap = 0
        for prev_bbox in prev_bboxes:

            if overlap == 0:

                # check lengthwise direction
                is_include_min_x_prev_bbox = current_bbox[0][0] in range(prev_bbox[0][0], prev_bbox[1][0])
                is_include_max_y_prev_bbox = current_bbox[1][1] in range(prev_bbox[0][1], prev_bbox[1][1])

                if is_include_min_x_prev_bbox and is_include_max_y_prev_bbox:

                    overlap = 1

                # check widthwise direction
                is_include_min_x_prev_bbox = current_bbox[1][1] in range(prev_bbox[0][0], prev_bbox[1][0])
                is_include_max_y_prev_bbox = current_bbox[0][0] in range(prev_bbox[0][1], prev_bbox[1][1])

                if is_include_min_x_prev_bbox and is_include_max_y_prev_bbox:

                    overlap = 1

        if overlap == 1:
            overlap_bboxes.append(current_bbox)

    return overlap_bboxes