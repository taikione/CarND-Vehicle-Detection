**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and append binned color feature and histogram of color to HOG feature vector.
* Train a  Linear SVM classifier.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Run the pipeline on a video stream (full project_video.mp4) and estimate a bounding box for vehicles detected.

[//]: # (Image References)
[vehicles]: ./examples/vehicles.png
[notvehicles]: ./examples/nonvehicles.png
[hog_feature]: ./examples/get_hog_feature.png
[serach_windows]: ./examples/serach_windows.png
[sliding_window]: ./examples/sliding_window.png
[scale_seach]: ./examples/scale_seach.png
[example_images]: ./examples/examples.png
[heatmaps]: ./examples/heatmaps.png
[compare_prev]: ./examples/compare_prev.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th code cell of VehicleDetection-LinearSVM.ipynb (`utilities.extract_features()`).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][vehicles]

![alt text][notvehicles]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][hog_feature]

In addition to these, I used the histogram of color and binned color feature.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters as below.

| orientations | pixels_per_cell | cells_per_block align |
|:-----------:|:------------:|:------------:|
| 9 | (8, 8)| 2 |
| 9 | (6, 6)| 2 |
| 7 | (6, 6)| 2 |

I investigated among these parameters and used the parameter with the highest SVM score.
I train and classified by SVM using these parameters, and the parameters with the highest prediction accuracy among them were used.

Finally, we decided on the following parameters.

`orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First, standardize HOG features, the histogram of color and binned color feature for SVM learning.(5th cell of `VehicleDetection-LinearSVM.ipynb`)
Next, I divided dataset for learning and testing.

I trained a linear SVM using the above parameters in the 6~9th cell of `VehicleDetection-LinearSVM.ipynb`. I also performed a grid search on SVM parameters.

Finally, I got a score of 99.04%.(9th cell)
```
9.23 Seconds to train SVC...
Test Accuracy of SVC =  0.9904
My SVC predicts:  [ 1.  1.  1.  1.  1.  0.  0.  1.  0.  1.]
For these 10 labels:  [ 1.  1.  1.  1.  1.  0.  0.  1.  0.  1.]
0.003 Seconds to predict 10 labels with SVC
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

I fixed the window size to 64 x 64 and 25% overlap, and the target area of the window search was done on the y-axis 400 to 600. In this execution, although the window size is fixed, the scale of the target area is adjusted.

![alt text][serach_windows]


#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

I searched on 5 target area scale applied window search (12th cell of `VehicleDetection-LinearSVM.ipynb`).

![alt text][scale_seach]

Finally, I used scale 1.1, and YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][example_images]

---

### Video Implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./result_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

##### Heatmap

I recorded the positions of positive detections in each frame of the video. From the positive detections, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. (14th cell of `VehicleDetection-LinearSVM.ipynb`)

Here's an example result showing the bounding boxes and heatmaps.

![alt text][heatmaps]

##### Compare previous frames
In order to detect a bounding box that was erroneously detected, bounding boxes of the current frame and previous frames were compared.

Assuming that correct bounding boxes are detected at similar positions over several tens of frames, I removed bounding boxes as false positives if the bounding box does not overlap with the previous frame. (`untile.py compare_prev_frame()`)

Here is an example.
![alt text][compare_prev]

Ultimately, I compare previous 11 frames(`CHECK_FRAME_RANGE` is the parameter for this).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In video implementation, there were things that could not be detected when the car goes far.
If you learn SVM by increasing variations of data, the result will be improved. Also, the result may be improved by create features and train SVM for each position (far and near position).
