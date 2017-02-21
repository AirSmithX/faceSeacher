import os
import cv2
import os
import sys
import dlib

def load_image_file_names(_file_dir):
    file_names = os.listdir(_file_dir)
    file_full_names = []

    print file_names

    if file_names:
        for name in file_names:
            #fill in the full names
            file_full_name = os.path.join(_file_dir, name)
            file_full_names.append(file_full_name)
    return file_full_names


def detect_face(_img, _detctor):
    dets = _detctor(_img, 1)
    if dets:
        return True
    else:
        return False