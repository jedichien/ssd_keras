#-*- coding: utf-8 -*-
import numpy as np
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from img_factory import ImgFactory
from ssd_utils import BBoxUtility

class Generator:
    
    def __init__(self, train, val, bbox_util, batch_size=32, image_size=(300, 300)):
        self.t_img_path, self.t_label, self.t_bbox = train
        self.v_img_path, self.v_label, self.v_bbox = val
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.image_size = image_size
        # some optional images processing coefficients
        self.saturation_var=0.5
        self.brightness_var=0.5
        self.contrast_var=0.5
        self.lighting_std=0.5
        self.hflip_prob=0.5
        self.vflip_prob=0.5
        # colorjitter sets
        self.imgFactory = ImgFactory(self.saturation_var, self.brightness_var, self.contrast_var,
                               self.lighting_std, self.hflip_prob, self.vflip_prob)
        self.color_jitter = [self.imgFactory.saturation, self.imgFactory.brightness, self.imgFactory.contrast]
        self.do_crop=True
        self.crop_area_range=[0.75, 1.0]
        self.aspect_ratio_range=[3./4., 4./3.]
        
    def random_sized_crop(self, img, targets):
        # area info
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h

        # calculate scale for image cropping
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] - self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area

        # calculate ratio for ceiling
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] - self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]

        # get minimist width and height between original and processed.
        w = np.round(np.sqrt(target_area * random_ratio))
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)

        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        # crop
        img = img[y:y+h, x:x+w]
        new_targets = []
        # recalculate bonding box
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets

    # generate data preprocess_img_array, and data shape with (batch_size, ground-truth-box{xmin, ymin, xmax, ymax} + NUM_CLASSES) 
    def generate(self, train=True):
        while True:
            if train:
                datazipped = zip(self.t_img_path, self.t_label, self.t_bbox)
                shuffle(datazipped)
            else:
                datazipped = zip(self.v_img_path, self.v_label, self.v_bbox)
                shuffle(datazipped)
            paths, labels, bboxs = zip(*datazipped)
            inputs = []
            targets = []
            for i in range(len(paths)):
                img = imread(paths[i]).astype('float32')
                bbox = np.array(bboxs[i]).copy()
                label = np.array(labels[i]).copy()
                if train and self.do_crop:
                    img, bbox = self.random_sized_crop(img, bbox)
                img = imresize(img, self.image_size).astype('float32')
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.imgFactory.lighting(img)
                    if self.hflip_prob > 0:
                        img, bbox = self.imgFactory.horizontal_flip(img, bbox)
                    if self.vflip_prob > 0:
                        img, bbox = self.imgFactory.vertical_flip(img, bbox)
                label = np.tile(label, (bbox.shape[0], 1))
                y = np.concatenate((bbox, label), axis=1)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets
