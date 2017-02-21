import cv2
import utils as ut
import dlib

import numpy as np

ENV_WIDTH = 2456
ENV_HEIGH = 1988

SAVE_PATH = "/home/air/washed_imgs"

FILE_DIR = "/media/air/000DC49500005230/activereforiencelearing/image_net"

def resize_file(_img):
    width = _img.shape[1]
    heigh = _img.shape[0]

    width_factor = ENV_WIDTH / float(width)
    heigh_factor = ENV_HEIGH / float(heigh)
    factor = max([width_factor, heigh_factor])

    img = cv2.resize(_img,(int(width*factor), int(heigh*factor)))

    img_new = np.zeros((ENV_HEIGH, ENV_WIDTH, 3))
    if width_factor < heigh_factor:
        img_new = img[:, (img.shape[1] - img_new.shape[1])/2:(img.shape[1] - img_new.shape[1])/2 + img_new.shape[1]]
    else:
        img_new = img[(img.shape[0] - img_new.shape[0])/2 : (img.shape[0] - img_new.shape[0])/2 + img_new.shape[0],:]
    return img_new
    # img_new = cv2.resize(img_new,(int(img_new.shape[1]/3),int(img_new.shape[0]/3)))

count = 0
detector = dlib.get_frontal_face_detector()
names = ut.load_image_file_names(FILE_DIR)
for name in names:
    print name
    img = cv2.imread(name)
    img_resized = resize_file(img)
    if ut.detect_face(img_resized,detector):
        count = count + 1
        save_path = SAVE_PATH + "/" + str(count) + ".jpg"
        print "save to :" + save_path
        cv2.imwrite(save_path, img_resized)



