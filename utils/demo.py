#-*- coding: utf-8 -*-
from scipy.misc import imread
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
"""
demonstration
"""
# img_path = './pics/fish-bike.jpg'
# img_path = './pics/my-test-motor.jpg'
# img_path='./pics/my-test-motor-man.jpg'
def demo(model, img_path, bbox_util):
    plt.rcParams['figure.figsize'] = (8, 8)
    plt.rcParams['image.interpolation'] = 'nearest'
    np.set_printoptions(suppress=True)
    # for imshow usage
    images = []
    images.append(imread(img_path))
    
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    
    inputs = []
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))
    
    results = model.predict(inputs, batch_size=1)
    results = bbox_util.detection_out(results)

    for i, img in enumerate(images):
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

        plt.imshow(img / 255.)
        currentAxis = plt.gca()

        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))

            score = top_conf[i]
            label = int(top_label_indices[i])
            #label_name = voc_classes[label-1]
            display_txt = '{:0.2f}, {}'.format(score, label)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

        plt.show()