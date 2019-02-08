import cv2
import os, time, sys, shutil
import numpy as np

PATH_DATA_FOLDER = './data/'

PATH_TRAIN_LABEL = PATH_DATA_FOLDER +  'train.txt'
PATH_TRAIN_VIDEO = PATH_DATA_FOLDER + 'train.mp4'
PATH_TRAIN_IMAGES_FOLDER = PATH_DATA_FOLDER +  'train_images/'

PATH_TEST_LABEL = PATH_DATA_FOLDER +  'test.txt'
PATH_TEST_VIDEO = PATH_DATA_FOLDER + 'test.mp4'
PATH_TEST_IMAGES_FOLDER = PATH_DATA_FOLDER +  'test_images/'

def convert_data(video_input_path, image_folder_path):

    if os.path.exists(image_folder_path):
        shutil.rmtree(image_folder_path)
    os.makedirs(image_folder_path)

    print("Converting video to frames: ", video_input_path)
    t1 = time.time()

    video_reader = cv2.VideoCapture(video_input_path)
    num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)

    count = 0
    while True:
        ret, next_frame = video_reader.read()
        if ret is False:
            break

        image_path_out = os.path.join(image_folder_path, str(count) + '.jpg')
        cv2.imwrite(image_path_out, next_frame)

        count += 1
        sys.stdout.write('\rprocessed frames: %d of %d' % (count, num_frames))


    t2 = time.time()
    video_reader.release()
    print(' Conversion completed !')
    print(' Time Taken:', (t2 - t1), 'seconds')
    return

if __name__ == '__main__':

    convert_data(PATH_TRAIN_VIDEO, PATH_TRAIN_IMAGES_FOLDER)
    convert_data(PATH_TEST_VIDEO, PATH_TEST_IMAGES_FOLDER)
