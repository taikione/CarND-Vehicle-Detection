from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
import numpy as np
import utilities


class Pipeline:
    def __init__(self, svc, X_scaler, scale, check_frame_range, threshold, orient, pix_per_cell, cell_per_block):
        self.previouse_bboxes_1 = []

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

    def __call__(self, image):

        # try multi scale window search
        detected_bboxes_1 = utilities.window_search(image, self.top, self.bottom, self.scale,
                                                    self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block)

        try:
            if len(self.previouse_bboxes_1) >= self.check_frame_range:
                s1_bboxes = self.previouse_bboxes_1.pop(0)

            else:
                s1_bboxes = self.previouse_bboxes_1[0]

            s1_compared_bboxes = [utilities.conpare_prev_frame(b, detected_bboxes_1) for b in self.previouse_bboxes_1]

            plot_bboxes = list(set(s1_bboxes).intersection(*s1_compared_bboxes))

        except:
            plot_bboxes = detected_bboxes_1

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        heat = utilities.add_heat(heat, plot_bboxes)

        heat = utilities.apply_threshold(heat, self.threshold)
        heatmap = np.clip(heat, 0, 255)

        labels = label(heatmap)
        draw_img = utilities.draw_labeled_bboxes(np.copy(image), labels)

        self.previouse_bboxes_1.append(detected_bboxes_1)

        return draw_img