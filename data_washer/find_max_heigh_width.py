import cv2
import utils as ut

FILE_DIR = "/home/air/washed_imgs"

def find_max_heigh_width():
    heigh_max = 0
    width_max = 0

    names = ut.load_image_file_names(FILE_DIR)

    for name in names:
        img = cv2.imread(name)
        if heigh_max < img.shape[0]:
            heigh_max = img.shape[0]
        if width_max < img.shape[1]:
            width_max = img.shape[1]

    print "heigh : ", heigh_max,"    width : ",width_max

find_max_heigh_width()