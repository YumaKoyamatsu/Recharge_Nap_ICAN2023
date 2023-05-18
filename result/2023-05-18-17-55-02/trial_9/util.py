import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import shutil
import datetime
import numpy as np
import random
import math
from tqdm import tqdm


#実行コードの保存
def save_running_code(running_script, file_name):
    # 実行中のスクリプトを新しいファイル名で保存
    shutil.copy(running_script, file_name)

def get_now_time():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))) # 日本時刻
    now = now.strftime('%Y-%m-%d-%H-%M-%S')
    return now

def create_batch(data, batch_size):
    """
    :param data: np.ndarray，入力データ
    :param batch_size: int，バッチサイズ
    """
    num_batches, mod = divmod(data.shape[0], batch_size)
    batched_data = np.split(data[: batch_size * num_batches], num_batches)
    if mod:
        batched_data.append(data[batch_size * num_batches:])

    return batched_data

def early_stop(avg_val_loss, best_val_loss, wait, patience=15, min_delta=0.0001):
    stop_flag = 0
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        wait = 0
        
    else:
        wait += 1
        if wait >= patience:
            print("Early Stopping triggered. Stopping training.")
            stop_flag = 1
    return wait, best_val_loss, stop_flag

#Data augmentation function
def random_rotation(img, angle_range=(-15, 15)):
    angle = random.uniform(*angle_range)
    angle = math.radians(angle)
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    h, w = img.shape
    c_y, c_x = h // 2, w // 2

    new_img = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            new_x = (x - c_x) * cos_val - (y - c_y) * sin_val + c_x
            new_y = (x - c_x) * sin_val + (y - c_y) * cos_val + c_y
            new_x, new_y = int(round(new_x)), int(round(new_y))
            if 0 <= new_x < w and 0 <= new_y < h:
                new_img[y, x] = img[new_y, new_x]
    return new_img

def random_translation(img, shift_range=(-2, 2)):
    shift_x = random.randint(*shift_range)
    shift_y = random.randint(*shift_range)
    return np.roll(img, (shift_x, shift_y), axis=(0, 1))

def random_scaling(img, scale_range=(0.9, 1.1)):
    scale = random.uniform(*scale_range)
    h, w = img.shape
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    scaled_img = np.zeros((new_h, new_w))
    
    for y in range(new_h):
        for x in range(new_w):
            orig_x, orig_y = int(x / scale), int(y / scale)
            if 0 <= orig_x < w and 0 <= orig_y < h:
                scaled_img[y, x] = img[orig_y, orig_x]
    
    if scale > 1.0:
        start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
        return scaled_img[start_h:start_h+h, start_w:start_w+w]
    else:
        pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
        pad_img = np.zeros((h, w))
        pad_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled_img
        return pad_img

def random_horizontal_flip(img, p=0.5):
    if random.random() < p:
        return np.fliplr(img)
    return img
def random_vertical_flip(img, p=0.5):
    if random.random() < p:
        return np.flipud(img)
    return img

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def random_pixel_removal(img):
    # 画像の形状を取得
    height, width = img.shape
    # 削除する画素数をランダムに設定（1～100まで）
    num_pixels_to_remove = np.random.randint(1, 101)
    for _ in range(num_pixels_to_remove):
        # ランダムな位置を選択
        rand_y = np.random.randint(0, height)
        rand_x = np.random.randint(0, width)
        # その位置の画素を黒にする
        img[rand_y, rand_x] = 0
    return img

def DataAugmentation(x, y):
    print("data_augmentation...")
    x_augmented = []
    y_augmented = []
    
    for img, label in zip(tqdm(x), y):
        # Keep the original image
        x_augmented.append(img)
        y_augmented.append(label)

        # Apply augmentation
        #img_rotated = random_rotation(img)
        img_translated = random_translation(img)
        img_scaled = random_scaling(img)
        img_flipped_lr = random_horizontal_flip(img)
        #img_flipped_ud = random_vertical_flip(img)
        #img_noise = add_gaussian_noise(img)
        #img_remove = random_pixel_removal(img)

        # Add augmented images
        x_augmented.extend([img_translated, img_scaled, img_flipped_lr])
        y_augmented.extend([label] * 3)
        
    x_augmented = np.array(x_augmented)
    y_augmented = np.array(y_augmented)
        
    return x_augmented, y_augmented