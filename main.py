import glob
import pprint
import numpy as np
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from moviepy.editor import VideoFileClip

import utilities
import vehicleDetection

def main(input_fname, output_fname):

    PIX_PER_CELL = 8
    CELL_PER_BLOCK = 2
    ORIENT = 9
    CHECK_FRAME_RANGE = 11
    THRESHOLD = 0
    TARGET_SCALE = 1.1

    # Divide up into cars and notcars
    images = glob.glob('data/*/*/*.png')
    vehicles = []
    notvehicles = []

    for image in images:
        if 'non-vehicles' in image:
            notvehicles.append(image)
        elif 'vehicles' in image:
            vehicles.append(image)
        else:
            print("error")

    car_features = utilities.extract_features(vehicles, orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK)

    notcar_features = utilities.extract_features(notvehicles, orient=ORIENT, pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    print("Fit a per-column scaler")
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    np.random.seed(10)
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)


    # Refer to VehicleDetection-LinearSVM.ipynb
    svc = LinearSVC(C=0.0004, loss='hinge', penalty='l2')

    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()

    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    pipeline = vehicleDetection.Pipeline(svc, X_scaler, TARGET_SCALE, CHECK_FRAME_RANGE, THRESHOLD,
                                        ORIENT, PIX_PER_CELL, CELL_PER_BLOCK)

    def pipe_image(image):
        return pipeline(image)

    clip1 = VideoFileClip(input_fname)
    white_clip = clip1.fl_image(pipe_image)
    white_clip.write_videofile(output_fname, audio=False)


if __name__ == "__main__":
    input_name = "project_video.mp4"
    output_name = "result_project_video_scale11_f11.mp4"

    main(input_fname=input_name, output_fname=output_name)