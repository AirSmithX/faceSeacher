import cv2
import os
import sys
import dlib
import utils as ut

FILE_DIR = "/media/air/000DC49500005230/activereforiencelearing/image_net"
SAVE_PATH = "/home/air/washed_imgs"



def wash_datas():
    file_names = ut.load_image_file_names(FILE_DIR)


    detector = dlib.get_frontal_face_detector()


    count = 0
    for name in file_names:
        img = cv2.imread(name)
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if ut.detect_face(img_RGB, detector):
            count = count + 1
            save_path = SAVE_PATH + "/" + str(count) + ".jpg"
            print "save to :" + save_path
            cv2.imwrite(save_path, img)
        else:
            pass


wash_datas()







