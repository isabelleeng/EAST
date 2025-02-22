import cv2
import time
import os
import argparse
import numpy as np
import tensorflow as tf
from timeit import default_timer
from keras.models import model_from_json
from nms import non_max_suppression
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, default='data\\test')
parser.add_argument('--video_path', type=str, default='')
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--model_path', type=str, default='model\\model.json')
parser.add_argument('--weights_path', type=str, default='model\\weights-60.h5')
parser.add_argument('--output_dir', type=str, default='tmp\\eval\\results')
parser.add_argument('--frame_no', type=int, default=0) # the frame number where evaluation starts or continues, default 0

FLAGS = parser.parse_args()

from model import *
from data_processor import restore_rectangle

def load_model():
    json_file = open(os.path.join('\\'.join(FLAGS.model_path.split('\\')[0:-1]), 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'tf': tf, 'RESIZE_FACTOR': RESIZE_FACTOR})
    model.load_weights(FLAGS.weights_path)
    return model

def get_file_name(video_path, frame_no):
    video_name = video_path.split('\\')
    video_name = video_name[-1].split('.')[0]
    file_name = f'{video_name}_frame_{frame_no}'
    return file_name

def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def detect(frame, score_map, geo_map, timer, score_map_thresh=0.99, box_thresh=0.1, nms_thres=0.3):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start

    # nms part - remove overlapping boxes
    start = time.time()
    boxes, accuracy = non_max_suppression(boxes.astype(np.float64), nms_thres, boxes[:, 8])
    timer['nms'] = time.time() - start

    if len(boxes) == 0:
        return None, None, timer

    return boxes, accuracy, timer

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    while True:
        video_path = input("Insert video path: ")
        if '"' in video_path:
                video_path = video_path.translate({ord('"'): None})
        validPath = os.path.exists(video_path)
        file_extension = video_path.split('.')[-1]
        if validPath == True and file_extension == "mp4":
            break
        else:
            print('Not a valid path. Try again.')

    # video = FLAGS.video_path
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    model = load_model()
    
    print('Scanning...')
    timeStart = default_timer()
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, FLAGS.frame_no)
        success, img_file = cap.read()
        if not success:
            cap.release()
            break
        
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        formatted_time = str(datetime.timedelta(milliseconds=timestamp))

        img = img_file[:, :, ::-1]

        img_resized, (ratio_h, ratio_w) = resize_image(img)
        img_resized = (img_resized / 127.5) - 1

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()

        score_map, geo_map = model.predict(img_resized[np.newaxis, :, :, :])
        timer['net'] = time.time() - start

        boxes, score, timer = detect(FLAGS.frame_no, score_map=score_map, geo_map=geo_map, timer=timer)
       
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes = boxes.astype(np.float32)
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        # save to file
        if boxes is not None:
            file_name = get_file_name(FLAGS.video_path, FLAGS.frame_no)
            res_file = os.path.join(
                FLAGS.output_dir,
                '{}.txt'.format(file_name))

            with open(res_file, 'w') as f:
                f.write(f'Timestamp: {formatted_time}\r\n')
                for box in boxes:
                    # to avoid submitting errors
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                    ))
                    cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=2)
                       
            cv2.putText(img[:, :, ::-1], formatted_time, org=(100,100), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2 , color=(255, 255, 0), thickness=2)
            img_path = os.path.join(FLAGS.output_dir, f'{file_name}.jpg')
            cv2.imwrite(img_path, img[:, :, ::-1])
            print('Saved frame to {}:\n (predict {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms)'.format(
                img_path, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))
        FLAGS.frame_no += fps
    
    timeEnd = default_timer()
    timer_seconds = timeEnd-timeStart
    totalTime = datetime.timedelta(seconds=timer_seconds)
    print(f'Evaluation completed!\nTotal time: {totalTime}')

if __name__ == '__main__':
    main()