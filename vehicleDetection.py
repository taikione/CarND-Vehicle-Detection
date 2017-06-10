from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
import numpy as np
import utilities


class Pipeline:
    def __init__(self, svc, X_scaler, scale, check_frame_range, threshold, orient, pix_per_cell, cell_per_block):

        self.svc = svc
        self.X_scaler = X_scaler
        self.scale = scale
        self.check_frame_range = check_frame_range
        self.threshold = threshold
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.top = 400
        self.bottom = 600

        self.previouse_bboxes_1 = []


    def __call__(self, image):

        detected_bboxes = utilities.window_search(image, self.top, self.bottom, self.scale,
                                                    self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block)

        if len(detected_bboxes) < 5:
            self.threshold = 0

        if len(detected_bboxes) > 0:

            if len(self.previouse_bboxes_1) == 0:

                plot_bboxes = detected_bboxes

            else:

                if len(self.previouse_bboxes_1) >= self.check_frame_range:

                    s1_bboxes = self.previouse_bboxes_1.pop(0)

                else:

                    s1_bboxes = self.previouse_bboxes_1[0]

                # Extract bounding boxes overlapped with previouse_bboxes
                s1_compared_bboxes = [utilities.compare_prev_frame(b, detected_bboxes) for b in self.previouse_bboxes_1]

                plot_bboxes = list(set(s1_bboxes).intersection(*s1_compared_bboxes))

            heat = np.zeros_like(image[:, :, 0]).astype(np.float)

            heat = utilities.add_heat(heat, plot_bboxes)

            heat = utilities.apply_threshold(heat, self.threshold)
            heatmap = np.clip(heat, 0, 255)

            labels = label(heatmap)
            draw_img = utilities.draw_labeled_bboxes(np.copy(image), labels)

            self.previouse_bboxes_1.append(detected_bboxes)

        else:
            draw_img = image

        return draw_img