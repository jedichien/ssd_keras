#-*- coding: utf-8 -*-
import numpy as np
"""
Image factory for generate_data usage, which main propose is to diversify datasets.
For colors we have `grayscale` , `saturation` , `brightness` , `contrast` and `lighting`
For flip we have `horizontal_flip` , `vertical_flip`
"""
class ImgFactory:
    def __init__(self, 
                 saturation_var, 
                 brightness_var, 
                 contrast_var, 
                 lighting_std, 
                 hflip_prob, 
                 vflip_prob):
        self.saturation_var = saturation_var
        self.brightness_var = brightness_var
        self.contrast_var = contrast_var
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob

    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1-alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1-alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    def horizontal_flip(self, img, y):
        y = np.array(y)
        if np.random.random() < self.hflip_prob:
            img = img[:,::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        y = np.array(y)
        if np.random.random() < self.vertical_flip:
            img = img[::-1]
            y[:,[1, 3]] = 1 - y[:, [3, 1]]
        return img, y
