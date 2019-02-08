from model3 import CNNModel
import cv2
import sys
import time
import numpy as np
from frames_to_opticalFlow import convertToOptical


PATH_DATA_FOLDER = './data/'
PATH_TEST_LABEL = PATH_DATA_FOLDER +  'test.txt'
PATH_TEST_VIDEO = PATH_DATA_FOLDER + 'test.mp4'
PATH_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'test_output.mp4'
PATH_TEST_IMAGES_FOLDER = PATH_DATA_FOLDER +  'test_images/'
PATH_TEST_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'test_images_flow/'

TYPE_FLOW_PRECOMPUTED = 0
TYPE_ORIGINAL = 1


MODEL_NAME = 'CNNModel_flow_3'

PRE_TRAINED_WEIGHTS = './best'+MODEL_NAME


def predict_from_video(video_input_path, video_output_path):
    predicted_labels = []

    video_reader = cv2.VideoCapture(video_input_path)

    num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = 0x00000021
    video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, frame_size)

    t1 = time.time()
    ret, prev_frame = video_reader.read()
    hsv = np.zeros_like(prev_frame)

    video_writer.write(prev_frame)

    predicted_labels.append(0.0)


    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,50)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    count =0
    while True:
        ret, next_frame = video_reader.read()
        if ret is False or count>500:
            break

        flow_image_bgr = convertToOptical(prev_frame, next_frame)

        curr_image = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
        combined_image = 0.3*curr_image + 1.5*flow_image_bgr

        combined_image = cv2.resize(combined_image, (0,0), fx=0.5, fy=0.5)

        combined_image = combined_image.reshape(1, combined_image.shape[0], combined_image.shape[1], combined_image.shape[2])

        prediction = model.predict(combined_image)

        predicted_labels.append(prediction[0][0])

        cv2.putText(next_frame, str(prediction[0][0]), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)

        # video_writer.write(combined_image)
        video_writer.write(next_frame)

        prev_frame = next_frame

        count +=1
        sys.stdout.write('\rprocessed frames: %d of %d' % (count, num_frames))


    t2 = time.time()
    video_reader.release()
    video_writer.release()
    print(' Prediction completed !')
    print(' Time Taken:', (t2 - t1), 'seconds')

    return predicted_labels



if __name__ == '__main__':

    model = CNNModel()
    model.load_weights(PRE_TRAINED_WEIGHTS)

    print('Testing model...')
    predicted_labels = predict_from_video(PATH_TEST_VIDEO,  PATH_TEST_VIDEO_OUTPUT)

    with open(PATH_TEST_LABEL, mode="w") as outfile:
        for label in predicted_labels:
            outfile.write("%s\n" % str(label))
