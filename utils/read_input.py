#-*- coding: utf-8 -*-
from protos import string_int_label_map_pb2
import tensorflow as tf
from google.protobuf import text_format
import logging
import random
from lxml import etree
import io
import PIL.Image
import hashlib
import numpy as np
import os
import re

def _get_label_map_dict(label_map_path):
    """load label_map"""
    def load_labelmap(path):
        with tf.gfile.GFile(path, 'r') as fid:
            label_map_string = fid.read()
            label_map = string_int_label_map_pb2.StringIntLabelMap()
            try:
                text_format.Merge(label_map_string, label_map)
            except text_format.ParseError:
                label_map.ParseFromString(label_map_string)
        return label_map
    
    label_map = load_labelmap(label_map_path)
    label_map_dict = {}
    for item in label_map.item:
        label_map_dict[item.name.lower()] = item.id
        
    return label_map_dict

"""get filename list"""
def _read_examples_list(path):
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(' ')[0] for line in lines]

"""parse xml to dict"""
def _recursive_parse_xml_to_dict(xml):
    if len(xml) == 0:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = _recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

"""get filename"""
def _get_class_name_from_filename(file_name):
    match = re.match(r'([0-9]+\.jpg)', file_name, re.I)
    return match.groups()[0]

"""convert xml derived dict to tf.Example proto."""
# generate feature dict for storing as TFRecords.
def _dict_to_tf_example(data, label_map_dict, image_subdirectory, img_shape, num_cls):
    # image
    img_path = os.path.join(image_subdirectory, data['filename'])
    #with tf.gfile.GFile(img_path, 'rb') as fid:
    #    encoded_jpg = fid.read()
    #encoded_jpg_io = io.BytesIO(encoded_jpg)
    #image = PIL.Image.open(encoded_jpg_io)
    #if image.format != 'JPEG':
    #    raise ValueError('Image format not JPEG')
    #image = image.resize(img_shape, PIL.Image.BILINEAR)
    
    # image dimension
    width = int(data['size']['width'])
    height = int(data['size']['height'])
  
    # left-top corner
    xmins = []
    ymins = []
    # right-bottom corner
    xmaxs = []
    ymaxs= []
    # others
    classes = np.zeros(num_cls)
    classes_text = []
    
    for obj in data['object']:
        xmin = float(obj['bndbox']['xmin'])
        xmax = float(obj['bndbox']['xmax'])
        ymin = float(obj['bndbox']['ymin'])
        ymax = float(obj['bndbox']['ymax'])
        # use ratio
        xmins.append(xmin / width)
        ymins.append(ymin / height)
        xmaxs.append(xmax / width)
        ymaxs.append(ymax / height)
        # others
        class_name = obj['name']
        classes_text.append(class_name.encode('utf8'))
        classes[label_map_dict[class_name]-1] = 1.0
    xmins = np.asarray(xmins)
    ymins = np.asarray(ymins)
    xmaxs = np.asarray(xmaxs)
    ymaxs = np.asarray(ymaxs)
    
    return {
        'bbox': zip(xmins, ymins, xmaxs, ymaxs),
        'image': img_path,
        'label': classes,
        'class_text': np.asarray(classes_text)
    }

def _build_dataset(examples, annotations_dir, label_map_dict, image_dir, img_shape, num_cls):
    input_imgs = []
    input_labels = []
    input_bbox = []
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('on image %d of %d', idx, len(examples))
        # image feature(including bounding-box, classes etc.)
        xml_path = os.path.join(annotations_dir, example + '.xml')
        if not os.path.exists(xml_path):
            logging.warning('Could not find %s, ignoring example.', xml_path)
            continue
        with tf.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = _recursive_parse_xml_to_dict(xml)['annotation']   
        try:
            tf_example = _dict_to_tf_example(data, label_map_dict, image_dir, img_shape, num_cls)
            input_imgs.append(tf_example['image'])
            input_labels.append(tf_example['label'])
            input_bbox.append(tf_example['bbox'])
        except ValueError:
            logging.warning('Invalid examples: %s, ignoring.', xml_path)
    return np.asarray(input_imgs), np.asarray(input_labels), np.asarray(input_bbox)

def read_input(data_dir='data/VOC2007', label_map_filename='pascal.pbtxt', train_data_ratio=0.7, img_shape=(300, 300), num_cls=20):
    label_map_path = os.path.join(data_dir, label_map_filename)
    image_dir = os.path.join(data_dir, 'JPEGImages')
    annotations_dir = os.path.join(data_dir, 'Annotations')
    examples_path = os.path.join(data_dir, 'ImageSets', 'Main', 'trainval.txt')
    examples_list = _read_examples_list(examples_path)

    label_map_dict = _get_label_map_dict(label_map_path)

    logging.info('Reading from dataset.')
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(train_data_ratio * num_examples)

    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    
    logging.info('%d trai and %d val examples.', 
                 len(train_examples), len(val_examples))
    
    train_dataset = _build_dataset(train_examples, annotations_dir, label_map_dict, image_dir, img_shape, num_cls)
    val_dataset = _build_dataset(val_examples, annotations_dir, label_map_dict, image_dir, img_shape, num_cls)
    
    return train_dataset, val_dataset
