import numpy as np
import cv2


class getLaneLine:
    """
    Class for Detecting Lane Lines, curvature and vehicle position
    """

    def __init__(self):

        self.left_lane_indices = []
        self.right_lane_indices = []

        # polynomial coefficients for lane lines
        self.left_fit = []
        self.right_fit = []


    def sliding_window_search(self, binary_warped, nwindows=9, margin=100, minpix=50):
        """
        Give a preprocessed image(binalize and perspective transform),
        compute polynomial coefficients of lane lines
        """

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if len(self.left_fit) == 0 and len(self.right_fit) == 0:

            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Set height of windows
            window_height = np.int(binary_warped.shape[0]/nwindows)

            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                # Draw the windows on the visualization image
                # cv2.rectangle(output_image,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
                # cv2.rectangle(output_image,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

                # Identify the nonzero pixels in x and y within the window
                good_left_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                    (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                    (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                # Append these indices to the lists
                self.left_lane_indices.append(good_left_indices)
                self.right_lane_indices.append(good_right_indices)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_indices) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_indices]))
                if len(good_right_indices) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_indices]))

            # Concatenate the arrays of indices
            self.left_lane_indices = np.concatenate(self.left_lane_indices)
            self.right_lane_indices = np.concatenate(self.right_lane_indices)

            # Extract left and right line pixel positions
            leftx = nonzerox[self.left_lane_indices]
            rightx = nonzerox[self.right_lane_indices]
            lefty = nonzeroy[self.left_lane_indices]
            righty = nonzeroy[self.right_lane_indices]

            # Fit a second order polynomial to each
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)

        # If we got lanelines in previous frame,
        # search lanelines based on previous polynomial coefficients
        elif len(self.left_fit) > 0 and len(self.right_fit) > 0:

            self.left_lane_indices = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) &
                                (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))

            self.right_lane_indices = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) &
                                (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))

            # Again, extract left and right line pixel positions
            leftx = nonzerox[self.left_lane_indices]
            rightx = nonzerox[self.right_lane_indices]
            lefty = nonzeroy[self.left_lane_indices]
            righty = nonzeroy[self.right_lane_indices]

            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)


    def get_plotted_lane_lines_binalized_image(self, binary_warped, margin=10):
        """
        Give a preprocessed image, return a plotted laneline image using left_fit and right_fit
        """

        # Create an output image to draw on and visualize the result
        output_image = np.zeros_like(binary_warped)
        output_image = np.dstack((output_image, output_image, output_image))
        window_image = output_image.copy()

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Plot the lane lines (left: red, right; blue)
        output_image[nonzeroy[self.left_lane_indices], nonzerox[self.left_lane_indices]] = [255, 0, 0]
        output_image[nonzeroy[self.right_lane_indices], nonzerox[self.right_lane_indices]] = [0, 0, 255]

        # Ganerate some data to draw the lane
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        # left_line_points = np.hstack((left_line_window1, left_line_window2))
        # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        # right_line_points = np.hstack((right_line_window1, right_line_window2))

        left_line_window = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        right_line_window = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])

        driving_points = np.hstack((left_line_window, right_line_window))

        cv2.fillPoly(window_image, np.int_([driving_points]), (0,255, 0))

        return cv2.addWeighted(output_image, 1, window_image, 0.3, 0)


    def determine_curvature(self, binary_warped):
        """
        Determine the curvature of center of lane lines
        """

        # Generate some data to represent lane-line height pixels
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 27.4/binary_warped.shape[0] # meters per pixel in y dimension, lane lines length / birds eye viewed image height
        xm_per_pix = 3.7/580 # meters per pixel in x dimension, lane lines width / lane lines width of birds eye viewed image

        # Fit new polynomials to x,y in world space
        mean_fit = np.mean([self.left_fit, self.right_fit], axis=0)
        left_and_right_fitx = mean_fit[0]*ploty**2 + mean_fit[1]*ploty + mean_fit[2]

        left_and_right_fit_cr = np.polyfit(ploty*ym_per_pix, left_and_right_fitx*xm_per_pix, 2)

        # Calculate the new radius of curvature
        center_curvature_radius = ((1 + (2*left_and_right_fit_cr[0]*y_eval*ym_per_pix + left_and_right_fit_cr[1])**2)**1.5) / np.absolute(2*left_and_right_fit_cr[0])

        return center_curvature_radius


    def get_vehicle_position(self):
        """
        Compute vehicle position with respect to center.
        if vehicle position value is negative, vehicle is right of center.
        :return float
        """

        left_x0 = self.left_fit[0]*720**2 + self.left_fit[1]*720 + self.left_fit[2]
        right_x0 = self.right_fit[0]*720**2 + self.right_fit[1]*720 + self.right_fit[2]

        lane_lines_center = np.mean([left_x0, right_x0])
        vehicle_position_center = 640

        xm_per_pix = 3.7/580

        return (lane_lines_center - vehicle_position_center)*xm_per_pix