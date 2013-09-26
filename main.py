# -*- coding: utf-8 -*-
'''
Created on 21.09.2013

@author: Mark
'''

import time
import numpy as np
import cv2

DEBUG = True




class OpenCVWindow(object):
    """
    Klass mis hõlmab endas ühe OpenCV nimelise akna avamist, sulgemist ning sellel pildi näitamist
    """
    def __init__(self, name, fullscreen = False):
        self.name = name
        self.fullscreen = fullscreen
    def __enter__(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
        if DEBUG:
            print("Opening window: " + str(self.name))
        return self
    def __exit__(self, type, value, traceback):
        cv2.destroyWindow(self.name)
        if DEBUG:
            print("Closing window: " + str(self.name))
    def disp_frame(self, frame):
        cv2.imshow(self.name, frame)

class LoomadWindow(OpenCVWindow):
    animals = ("Inimene", "Koer", "Kärbes", "Kuldkala")
    """
    Loomade aken laiendab tavalist OpenCV akent erinevate filtritega, mis imiteerivad loomade nägemist
    """
    def __init__(self, name, fullscreen = False):
        OpenCVWindow.__init__(self, name, fullscreen)
        self.curr_animal_nr = 0  # Hetkelooma number

        self.icon_bg = cv2.imread("image_bg.png", cv2.CV_LOAD_IMAGE_UNCHANGED)
        self.icon_human = None
        self.icon_dog = cv2.imread("koer.png", cv2.CV_LOAD_IMAGE_UNCHANGED)
        self.icon_fly = cv2.imread("karbes.png", cv2.CV_LOAD_IMAGE_UNCHANGED)
        self.icon_goldfish = cv2.imread("kuldkala.png", cv2.CV_LOAD_IMAGE_UNCHANGED)

        self.karbes_grid_overlay = cv2.imread("karbes_grid.png", cv2.CV_LOAD_IMAGE_UNCHANGED)
        self.fly_grid_alpha = None

    def disp_frame(self, frame):
        """Meetod, mis kuvab pildi loomale vastava efekti ja ikooniga"""
        frame = cv2.resize(frame, (0, 0), fx = 2, fy = 2)  # Kuna kaamerapilt ei ole piisava resolutsiooniga, et ikoon kena jääks, skaleerime kaamerapilti üles

        animal = LoomadWindow.animals[self.curr_animal_nr]
        icon = None
        if animal == "Inimene":
            icon = self.icon_human
        elif animal == "Koer":
            frame = self.mod_dog(frame)
            icon = self.icon_dog
        elif animal == "Kärbes":
            frame = self.mod_fly(frame)
            icon = self.icon_fly
        elif animal == "Kuldkala":
            frame = self.mod_goldfish(frame)
            icon = self.icon_goldfish

        overlay_x = 110
        overlay_y = 110
        self.overlay(frame, self.icon_bg, overlay_x, overlay_y)
        if icon is not None:
            self.overlay(frame, icon, overlay_x, overlay_y)

        OpenCVWindow.disp_frame(self, frame)

    def nextAnimal(self):
        self.curr_animal_nr = (self.curr_animal_nr + 1) % len(LoomadWindow.animals)

    def mod_dog(self, frame):
        # Imitate dog's color-vision
        frame = cv2.transform(frame, np.array([ [ 1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [ 0.0, 0.5, 0.5] ]))
        # imitate dog's visual acuity
        frame = cv2.GaussianBlur(frame, (9, 9), 5)
        return frame

    def mod_fly(self, frame):
        # Black and white vision
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 5)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Fly grid
        self.overlay_alpha_only(frame, self.karbes_grid_overlay, frame.shape[1] / 2, frame.shape[0] / 2, 0.7)
        return frame
    def mod_goldfish(self, frame):
        frame = cv2.transform(frame, np.array([ [ 0.4, 0.0, 0.0], [0.0, 0.6, 0.0], [ 0.0, 0.0, 2.0] ]))
        return frame

    def calc_from_to(self, frame, overlay, x_pos, y_pos):
        # print("frame shape:", frame.shape)
        # print("overlay shape:", overlay.shape)
        # print("position:", x_pos, y_pos)

        f_x_from = max(x_pos - overlay.shape[1] / 2, 0)
        f_x_to = min(x_pos + overlay.shape[1] / 2 + (overlay.shape[1] % 2), frame.shape[1])
        f_y_from = max(y_pos - overlay.shape[0] / 2, 0)
        f_y_to = min(y_pos + overlay.shape[0] / 2 + (overlay.shape[0] % 2), frame.shape[0])
        # print("frame from-to", f_x_from, f_x_to, f_y_from, f_y_to)


        o_x_from = overlay.shape[1] / 2 - (x_pos - f_x_from)
        o_x_to = overlay.shape[1] - (overlay.shape[1] / 2 + (overlay.shape[1] % 2) - (f_x_to - x_pos))
        o_y_from = overlay.shape[0] / 2 - (y_pos - f_y_from)
        o_y_to = overlay.shape[0] - (overlay.shape[0] / 2 + (overlay.shape[0] % 2) - (f_y_to - y_pos))
        # print("overlay from-to", o_x_from, o_x_to, o_y_from, o_y_to)
        return (f_x_from, f_x_to, f_y_from, f_y_to, o_x_from, o_x_to, o_y_from, o_y_to);

    def overlay(self, frame, overlay, x_pos, y_pos):
        (f_x_from, f_x_to, f_y_from, f_y_to, o_x_from, o_x_to, o_y_from, o_y_to) = self.calc_from_to(frame, overlay, x_pos, y_pos)

        for c in range(0, 3):
            frame[f_y_from:f_y_to, f_x_from:f_x_to, c] = \
                overlay[o_y_from:o_y_to, o_x_from:o_x_to, c] * (overlay[o_y_from:o_y_to, o_x_from:o_x_to, 3] / 255.0) \
                + frame[f_y_from:f_y_to, f_x_from:f_x_to, c] * (1.0 - overlay[o_y_from:o_y_to, o_x_from:o_x_to, 3] / 255.0)

    def overlay_alpha_only(self, frame, overlay, x_pos, y_pos, opaque):
        if self.fly_grid_alpha == None:
            (f_x_from, f_x_to, f_y_from, f_y_to, o_x_from, o_x_to, o_y_from, o_y_to) = self.calc_from_to(frame, overlay, x_pos, y_pos)
            self.fly_grid_alpha = (1.0 - overlay[o_y_from:o_y_to, o_x_from:o_x_to, 3] / 255.0 * opaque)

        for c in range(0, 3):
            frame[:, :, c] *= self.fly_grid_alpha


class AnimalVision(object):
    def __init__(self):
        self.vc = cv2.VideoCapture(1)

    def run(self):
        print("Starting - Hello!")
        if self.vc.isOpened():  # try to get the first frame
            rval, frame = self.vc.read()
        else:
            raise Exception("Could not open camera!")

        timer = time.time()
        with LoomadWindow("Loomad", True) as loomadWindow:
            while rval:
                if (time.time() - timer) > 5:
                    timer = time.time()
                    loomadWindow.nextAnimal()

                # read from camera
                rval, frame = self.vc.read()
                # flip to imitate mirror
                frame = cv2.flip(frame, 1)

                # Display
                loomadWindow.disp_frame(frame)
                # Stop, if ESC pressed
                key = cv2.waitKey(20)
                if key == 27:
                    break

            else:
                raise Exception("Reading from camera failed!")
        print("Exiting - Good bye!")


if __name__ == '__main__':
    av = AnimalVision()
    av.run()
