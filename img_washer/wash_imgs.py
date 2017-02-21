import os
import dlib
import cv2



FILE_PATH = "/home/air/face_seacher_train_dir/washed_imgs_2"
SAVE_PATH = "/home/air/face_seacher_train_dir/washed_imgs/"


def GetFileList(FindPath):
    FileList=[]
    FileNames=os.listdir(FindPath)
    for fn in FileNames:
       fullfilename=os.path.join(FindPath,fn)
       FileList.append(fullfilename)
    return FileList


file_lists = GetFileList(FILE_PATH)
detector = dlib.get_frontal_face_detector()

for file in file_lists:
    img = cv2.imread(file)
    # assert it opened right
    img_RGB = img.copy()
    img_RGB = cv2.cvtColor(img_RGB,cv2.COLOR_BGR2RGB)
    dets = detector(img_RGB, 1)
    print file
    if dets :
        if len(dets) > 1:
            print "too many faces"
        else:
            face_aread = (dets[0].right() - dets[0].left()) * (dets[0].bottom() - dets[0].top())
            area = img.shape[0] * img.shape[1]
            percent = float(face_aread) / float(area)


            if percent < 0.06:
                tail = file.split('/')[-1]
                full_path = SAVE_PATH + tail
                cv2.imwrite(full_path, img)
                print "saved  success"
            else:
                print "too big face"
    else:
        print "but no face"



        # for i, det in enumerate(dets):
        #     print i
