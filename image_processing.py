import cv2
import numpy as np

def preprocess_image(img_bytes):
    contrast_img = change_contrast(img_bytes)
    padded_img = pad_image(contrast_img)
    color_image = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    return color_image

def change_contrast(img):
    img = np.uint8(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    new_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return new_img


def pad_image(img, desired_size=384):
    old_size = img.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1], new_size[0]))
    remain_w = desired_size - new_size[1]
    remain_h = desired_size - new_size[0]
    top, bottom = remain_h//2, remain_h-(remain_h//2)
    left, right = remain_w//2, remain_w-(remain_w//2)
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img