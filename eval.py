import cv2
import time
import os
import argparse
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from nms import non_max_suppression

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, default='data\\test')
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--model_path', type=str, default='tmp/model/model.json')
parser.add_argument('--weights_path', type=str, default='weights\\weights-60.h5')
parser.add_argument('--output_dir', type=str, default='tmp/eval/results/images')
FLAGS = parser.parse_args()

from model import *
from losses import dice_loss
from data_processor import restore_rectangle

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


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


def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.3):
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
    #print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    
    # nms part
    boxes = non_max_suppression(boxes.astype(np.float64), nms_thres, boxes[:, 8])

    if boxes.shape[0] == 0:
        return None

    return boxes


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

    #load trained model
    json_file = open(os.path.join('/'.join(FLAGS.model_path.split('/')[0:-1]), 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'tf': tf, 'RESIZE_FACTOR': RESIZE_FACTOR})
    model.load_weights(FLAGS.weights_path)

    img_list = get_images()
    print("Scanning...")
    for img_file in img_list:
        img = cv2.imread(img_file)[:, :, ::-1]
        start_time = time.time()
        img_resized, (ratio_h, ratio_w) = resize_image(img)

        img_resized = (img_resized / 127.5) - 1

        # feed image into model
        score_map, geo_map = model.predict(img_resized[np.newaxis, :, :, :])

        boxes = detect(score_map=score_map, geo_map=geo_map)
       
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes = boxes.astype(np.float32)
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        # save to file
        if boxes is not None:
            res_file = os.path.join(
                FLAGS.output_dir,
                '{}.txt'.format(
                    os.path.basename(img_file).split('.')[0]))

            # with open(res_file, 'w') as f:
        for box in boxes:
            # to avoid submitting errors
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                continue
            # f.write('{},{},{},{},{},{},{},{}\r\n'.format(
            #     box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
            # ))
            cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=2)

        img_path = os.path.join(FLAGS.output_dir, os.path.basename(img_file))
        cv2.imwrite(img_path, img[:, :, ::-1])
    duration = time.time() - start_time
    print('Time elapsed: {}'.format(duration))

if __name__ == '__main__':
    main()