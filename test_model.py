from model import CNNModel
import cv2
import sys
import time
import numpy as np
from frames_to_opticalFlow import convertToOptical
import matplotlib.pyplot as plt

PATH_DATA_FOLDER = './data/'
PATH_TEST_LABEL = PATH_DATA_FOLDER +  'test.txt'
PATH_TEST_VIDEO = PATH_DATA_FOLDER + 'test.mp4'
PATH_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'test_output.mp4'
PATH_COMBINED_TEST_VIDEO_OUTPUT = PATH_DATA_FOLDER + 'combined_test_output.mp4'
PATH_TEST_IMAGES_FOLDER = PATH_DATA_FOLDER +  'test_images/'
PATH_TEST_IMAGES_FLOW_FOLDER = PATH_DATA_FOLDER +  'test_images_flow/'

TYPE_FLOW_PRECOMPUTED = 0
TYPE_ORIGINAL = 1


MODEL_NAME = 'CNNModel_flow'
# MODEL_NAME = 'CNNModel_combined'

PRE_TRAINED_WEIGHTS = './best'+MODEL_NAME+'.h5'


def predict_from_video(video_input_path, original_video_output_path, combined_video_output_path):
    predicted_labels = []

    video_reader = cv2.VideoCapture(video_input_path)

    num_frames = video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_size = (int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = 0x00000021
    video_writer = cv2.VideoWriter(original_video_output_path, fourcc, fps, frame_size)
    video_writer_combined = cv2.VideoWriter(combined_video_output_path, fourcc, fps, frame_size)

    t1 = time.time()
    ret, prev_frame = video_reader.read()
    hsv = np.zeros_like(prev_frame)

    video_writer.write(prev_frame)

    predicted_labels.append(0.0)

    flow_image_bgr_prev1 =  np.zeros_like(prev_frame)
    flow_image_bgr_prev2 =  np.zeros_like(prev_frame)
    flow_image_bgr_prev3 =  np.zeros_like(prev_frame)
    flow_image_bgr_prev4 =  np.zeros_like(prev_frame)


    font                   = cv2.FONT_HERSHEY_SIMPLEX
    place = (50,50)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    count =0
    while True:
        ret, next_frame = video_reader.read()
        if ret is False:
            break

        flow_image_bgr_next = convertToOptical(prev_frame, next_frame)
        flow_image_bgr = (flow_image_bgr_prev1 + flow_image_bgr_prev2 +flow_image_bgr_prev3 +flow_image_bgr_prev4 + flow_image_bgr_next)/4

        curr_image = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)

        combined_image_save = 0.1*curr_image + flow_image_bgr

        #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
        combined_image = flow_image_bgr
        # combined_image = combined_image_save

        combined_image_test = cv2.normalize(combined_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # plt.imshow(combined_image)
        # plt.show()

        #CHOOSE IF WE WANT TO TEST WITH ONLY OPTICAL FLOW OR A COMBINATION OF VIDEO AND OPTICAL FLOW
        # combined_image_test = cv2.resize(combined_image, (0,0), fx=0.5, fy=0.5)
        combined_image_test = cv2.resize(combined_image_test, (0,0), fx=0.5, fy=0.5)

        combined_image_test = combined_image_test.reshape(1, combined_image_test.shape[0], combined_image_test.shape[1], combined_image_test.shape[2])

        prediction = model.predict(combined_image_test)

        predicted_labels.append(prediction[0][0])

        # print(combined_image.shape, np.mean(flow_image_bgr), prediction[0][0])

        cv2.putText(next_frame, str(prediction[0][0]), place, font, fontScale,fontColor,lineType)
        cv2.putText(combined_image_save, str(prediction[0][0]), place, font, fontScale,fontColor,lineType)

        video_writer.write(next_frame)
        video_writer_combined.write(combined_image_save.astype('uint8'))

        prev_frame = next_frame
        flow_image_bgr_prev4 = flow_image_bgr_prev3
        flow_image_bgr_prev3 = flow_image_bgr_prev2
        flow_image_bgr_prev2 = flow_image_bgr_prev1
        flow_image_bgr_prev1 = flow_image_bgr_next

        count +=1
        sys.stdout.write('\rprocessed frames: %d of %d' % (count, num_frames))


    t2 = time.time()
    video_reader.release()
    video_writer.release()
    video_writer_combined.release()
    print(' Prediction completed !')
    print(' Time Taken:', (t2 - t1), 'seconds')

    predicted_labels[0] = predicted_labels[1]
    return predicted_labels



if __name__ == '__main__':

    model = CNNModel()
    model.load_weights(PRE_TRAINED_WEIGHTS)

    print('Testing model...')
    predicted_labels = predict_from_video(PATH_TEST_VIDEO,  PATH_TEST_VIDEO_OUTPUT, PATH_COMBINED_TEST_VIDEO_OUTPUT)

    with open(PATH_TEST_LABEL, mode="w") as outfile:
        for label in predicted_labels:
            outfile.write("%s\n" % str(label))
