#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:28:21 2017
@author: Hriddhi Dey

This module contains the ApplyMakeup class.
"""

import itertools
import scipy.interpolate
import cv2
import numpy as np
from skimage import color
from  detect_features import DetectLandmarks


class ApplyMakeup(DetectLandmarks):
    """
    Class that handles application of color, and performs blending on image.

    Functions available for use:
        1. apply_lipstick: Applies lipstick on passed image of face.
        2. apply_liner: Applies black eyeliner on passed image of face.
    """

    def __init__(self):
        """ Initiator method for class """
        DetectLandmarks.__init__(self)
        self.red_l = 0
        self.green_l = 0
        self.blue_l = 0
        self.red_e = 0
        self.green_e = 0
        self.blue_e = 0
        self.image = 0
        self.width = 0
        self.height = 0
        self.im_copy = 0
        self.lip_x = []
        self.lip_y = []


    def __read_image(self, filename):
        """ Read image from path forwarded """
        self.image = cv2.imread(filename)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.im_copy = self.image.copy()
        self.height, self.width = self.image.shape[:2]


    def __draw_curve_up(self, points):
        """
        Draws a curve alone the given points by creating an interpolated path.
        And Traverse x poions of upper lip.
        """
        x_pts = []
        y_pts = []
        for point in points:
            x_pts.append(point[0])
            y_pts.append(point[1])
        curve = scipy.interpolate.interp1d(x_pts, y_pts, 'quadratic')
        x_traver = np.arange(x_pts[0], x_pts[-1]+1, 1)
        return curve, x_traver


    def __draw_curve_low(self, points):
        """
        Draws a curve alone the given points by creating an interpolated path.
        And Traverse x poions of lower lip.
        """
        x_pts = []
        y_pts = []
        for point in points:
            x_pts.append(point[0])
            y_pts.append(point[1])
        curve = scipy.interpolate.interp1d(x_pts, y_pts, 'cubic')
        x_traver = np.arange(x_pts[0], x_pts[-1]+1, 1)
        return curve, x_traver


    def __smoothen_color(self, outer, inner):
        """ Smoothens and blends colour applied between a set of outlines. """
        outer_curve = zip(outer[0], outer[1])
        inner_curve = zip(inner[0], inner[1])
        x_points = []
        y_points = []
        for point in outer_curve:
            x_points.append(point[0])
            y_points.append(point[1])
        for point in inner_curve:
            x_points.append(point[0])
            y_points.append(point[1])
        img_base = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(img_base, np.array(np.c_[x_points, y_points], dtype='int32'), 1)
        img_mask = cv2.GaussianBlur(img_base, (51, 51), 0)
        img_blur_3d = np.ndarray([self.height, self.width, 3], dtype='float')
        img_blur_3d[:, :, 0] = img_mask
        img_blur_3d[:, :, 1] = img_mask
        img_blur_3d[:, :, 2] = img_mask
        self.im_copy = (img_blur_3d * self.image + (1 - img_blur_3d) * self.im_copy).astype('uint8')


    def __draw_liner(self, eye, kind):
        """ Draws eyeliner. """
        eye_x = []
        eye_y = []
        x_points = []
        y_points = []
        for point in eye:
            x_points.append(int(point.split()[0]))
            y_points.append(int(point.split()[1]))
        curve = scipy.interpolate.interp1d(x_points, y_points, 'quadratic')
        for point in np.arange(x_points[0], x_points[len(x_points) - 1] + 1, 1):
            eye_x.append(point)
            eye_y.append(int(curve(point)))
        if kind == 'left':
            y_points[0] -= 1
            y_points[1] -= 1
            y_points[2] -= 1
            x_points[0] -= 5
            x_points[1] -= 1
            x_points[2] -= 1
            curve = scipy.interpolate.interp1d(x_points, y_points, 'quadratic')
            count = 0
            for point in np.arange(x_points[len(x_points) - 1], x_points[0], -1):
                count += 1
                eye_x.append(point)
                if count < (len(x_points) / 2):
                    eye_y.append(int(curve(point)))
                elif count < (2 * len(x_points) / 3):
                    eye_y.append(int(curve(point)) - 1)
                elif count < (4 * len(x_points) / 5):
                    eye_y.append(int(curve(point)) - 2)
                else:
                    eye_y.append(int(curve(point)) - 3)
        elif kind == 'right':
            x_points[3] += 5
            x_points[2] += 1
            x_points[1] += 1
            y_points[3] -= 1
            y_points[2] -= 1
            y_points[1] -= 1
            curve = scipy.interpolate.interp1d(x_points, y_points, 'quadratic')
            count = 0
            for point in np.arange(x_points[len(x_points) - 1], x_points[0], -1):
                count += 1
                eye_x.append(point)
                if count < (len(x_points) / 2):
                    eye_y.append(int(curve(point)))
                elif count < (2 * len(x_points) / 3):
                    eye_y.append(int(curve(point)) - 1)
                elif count < (4 * len(x_points) / 5):
                    eye_y.append(int(curve(point)) - 2)
                elif count:
                    eye_y.append(int(curve(point)) - 3)
        curve = zip(eye_x, eye_y)
        points = []
        for point in curve:
            points.append(np.array(point, dtype=np.int32))
        points = np.array(points, dtype=np.int32)
        self.red_e = int(self.red_e)
        self.green_e = int(self.green_e)
        self.blue_e = int(self.blue_e)
        cv2.fillPoly(self.im_copy, [points], (self.red_e, self.green_e, self.blue_e))
        return


    def __add_color(self, intensity):
        """ Adds base colour to all points on lips, at mentioned intensity. """
        val = color.rgb2lab(
            (self.image[self.lip_x, self.lip_y] / 255.)
            .reshape(len(self.lip_x), 1, 3)
        ).reshape(len(self.lip_x), 3)
        l_val, a_val, b_val = np.mean(val[:, 0]), np.mean(val[:, 1]), np.mean(val[:, 2])
        l1_val, a1_val, b1_val = color.rgb2lab(
            np.array(
                (self.red_l / 255., self.green_l / 255., self.blue_l / 255.)
                ).reshape(1, 1, 3)
            ).reshape(3,)
        l_final, a_final, b_final = (l1_val - l_val) * \
            intensity, (a1_val - a_val) * \
            intensity, (b1_val - b_val) * intensity
        val[:, 0] = np.clip(val[:, 0] + l_final, 0, 100)
        val[:, 1] = np.clip(val[:, 1] + a_final, -127, 128)
        val[:, 2] = np.clip(val[:, 2] + b_final, -127, 128)
        self.image[self.lip_x, self.lip_y] = color.lab2rgb(val.reshape(
            len(self.lip_x), 1, 3)).reshape(len(self.lip_x), 3) * 255


    def __get_points_lips(self, lips_points):
        """ Get the points for the lips. """
        uol = []
        uor = []
        uil = []
        uir = []
        lo = []
        li = []
        for i in range(0, 8, 2):
            uol.append([int(lips_points[i]), int(lips_points[i + 1])])
        for i in range(6, 14, 2):
            uor.append([int(lips_points[i]), int(lips_points[i + 1])])
        for i in range(12, 24, 2):
            lo.append([int(lips_points[i]), int(lips_points[i + 1])])
        lo.append([int(lips_points[0]), int(lips_points[1])])
        for i in range(24, 30, 2):
            uil.append([int(lips_points[i]), int(lips_points[i + 1])])
        for i in range(28, 34, 2):
            uir.append([int(lips_points[i]), int(lips_points[i + 1])])
        for i in range(32, 40, 2):
            li.append([int(lips_points[i]), int(lips_points[i + 1])])
        li.append([int(lips_points[24]), int(lips_points[25])])
        return uol, uor, uil, uir, lo, li


    def __get_curves_lips(self, uol, uor, uil, uir, lo, li):
        """ Get the outlines and x points of the lips. """
        uol_curve = self.__draw_curve_up(uol)
        uor_curve = self.__draw_curve_up(uor)
        uil_curve = self.__draw_curve_up(uil)
        uir_curve = self.__draw_curve_up(uir)
        lo_curve = self.__draw_curve_low(lo)
        li_curve = self.__draw_curve_low(li)
        return uol_curve, uor_curve, uil_curve, uir_curve, lo_curve, li_curve


    def __traverse_store(self, a, b, i):
        """ Traverse along y axis with given x poionts in two curves and store them in lip_x and lip_y. """
        a, b = np.around(a), np.around(b)
        self.lip_x.extend(np.arange(a, b, 1, dtype=np.int32).tolist())
        self.lip_y.extend((np.ones(int(b - a), dtype=np.int32) * i).tolist())


    def __get_whole_lips(self, uol_curve, uor_curve, uil_curve, uir_curve, lo_curve, li_curve):
        """ Get all points inside lips curves. """
        for i in range(int(uol_curve[1][0]), int(uil_curve[1][0] + 1)):
            self.__traverse_store(uol_curve[0](i), lo_curve[0](i) + 1, i)

        for i in range(int(uil_curve[1][0]), int(uol_curve[1][-1] + 1)):
            self.__traverse_store(uol_curve[0](i), uil_curve[0](i) + 1, i)
            self.__traverse_store(li_curve[0](i), lo_curve[0](i) + 1, i)

        for i in range(int(uir_curve[1][-1]), int(uor_curve[1][-1] + 1)):
            self.__traverse_store(uor_curve[0](i), lo_curve[0](i) + 1, i)

        for i in range(int(uir_curve[1][0]), int(uir_curve[1][-1] + 1)):
            self.__traverse_store(uor_curve[0](i), uir_curve[0](i) + 1, i)
            self.__traverse_store(li_curve[0](i), lo_curve[0](i) + 1, i)


    def __create_eye_liner(self, eyes_points):
        """ Apply eyeliner. """
        left_eye = eyes_points[0].split('\n')
        right_eye = eyes_points[1].split('\n')
        right_eye = right_eye[0:4]
        self.__draw_liner(left_eye, 'left')
        self.__draw_liner(right_eye, 'right')


    def apply_lipstick(self, filename, rlips, glips, blips):
        """
        Applies lipstick on an input image.
        ___________________________________
        Args:
            1. `filename (str)`: Path for stored input image file.
            2. `red (int)`: Red value of RGB colour code of lipstick shade.
            3. `blue (int)`: Blue value of RGB colour code of lipstick shade.
            4. `green (int)`: Green value of RGB colour code of lipstick shade.

        Returns:
            `filepath (str)` of the saved output file, with applied lipstick.

        """

        self.red_l = rlips
        self.green_l = glips
        self.blue_l = blips
        self.__read_image(filename)
        lips = self.get_lips(self.image)
        lips = list([point.split() for point in lips.split('\n')])
        lips_points = [item for sublist in lips for item in sublist]
        uol, uor, uil, uir, lo, li = self.__get_points_lips(lips_points)
        uol_curve, uor_curve, uil_curve, uir_curve, lo_curve, li_curve = self.__get_curves_lips(uol, uor, uil, uir, lo, li)
        self.__get_whole_lips(uol_curve, uor_curve, uil_curve, uir_curve, lo_curve, li_curve)
        self.__add_color(1)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        name = 'color_' + str(self.red_l) + '_' + str(self.green_l) + '_' + str(self.blue_l)
        file_name = 'output_' + name + '.jpg'
        cv2.imwrite(file_name, self.image)
        return file_name


    def apply_liner(self, filename):
        """
        Applies lipstick on an input image.
        ___________________________________
        Args:
            1. `filename (str)`: Path for stored input image file.

        Returns:
            `filepath (str)` of the saved output file, with applied lipstick.

        """
        self.__read_image(filename)
        liner = self.get_upper_eyelids(self.image)
        eyes_points = liner.split('\n\n')
        self.__create_eye_liner(eyes_points)
        self.im_copy = cv2.cvtColor(self.im_copy, cv2.COLOR_BGR2RGB)
        name = '_color_' + str(self.red_l) + '_' + str(self.green_l) + '_' + str(self.blue_l)
        file_name = 'output_' + name + '.jpg'
        cv2.imwrite(file_name, self.im_copy)
        return file_name
