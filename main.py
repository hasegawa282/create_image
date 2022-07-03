import argparse
import glob
import os

import cv2
import numpy as np
import random

max_level = 3

def resize_padding(image, max_width, max_height):
    min_ratio = min(max_width / image.shape[1], max_height / image.shape[0])
    resized_image = cv2.resize(image, None, fx=min_ratio, fy=min_ratio)
    vertical_padding_size = max_height - resized_image.shape[0]
    horizon_padding_size = max_width - resized_image.shape[1]
    padding_image = cv2.copyMakeBorder(resized_image, vertical_padding_size // 2, vertical_padding_size // 2,
                                       horizon_padding_size // 2, horizon_padding_size // 2,
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
    resize_padding_image = cv2.resize(padding_image, (max_width, max_height))
    return resize_padding_image


def write_char(image, level, color):
    return cv2.putText(image, str(level), (0, image.shape[0] // 6), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       min(image.shape[:2]) // 70, color, min(image.shape[:2]) // 50, cv2.LINE_AA)


def transparent(image, level, color):
    canvas_image = np.zeros(image.shape, np.uint8)
    canvas_image += np.asarray((color[2], color[1], color[0]), np.uint8)[::-1]
    level_image = cv2.addWeighted(image, 0.9, canvas_image, 0.1, 2.2)
    white_image = np.ones(image.shape, np.uint8) * 255
    return cv2.addWeighted(level_image, 1/max_level*level, white_image, 1/max_level*(max_level-level), 2.2)



def thumbnail(image_dict, col_num):
    image_list = []
    for level in range(1, max_level + 1):
        image_list.extend(image_dict[level])

    row_num = len(image_list) // col_num
    if len(image_list) % col_num != 0:
        row_num += 1
    org_shape = image_list[0].shape
    canvas_image = np.ones((org_shape[0] * row_num, org_shape[1] * col_num, 3), np.uint8) * 255
    image_list = image_list[::-1]
    index = 0
    for row in range(row_num):
        for col in range(col_num):
            canvas_image[row * org_shape[0]:(row + 1) * org_shape[0], col * org_shape[1]:(col + 1) * org_shape[1], :] = \
            image_list[index] if index < len(image_list) else np.ones(org_shape, np.uint8)*255
            index += 1
    return canvas_image


def main(image_dir_path, col_num=5, width=160, height=160):
    image_dict = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    for level in range(1, 8):
        image_path_list = glob.glob(os.path.join(image_dir_path, str(level), '*.*'))
        color = (int(255 // max_level * (max_level - level)), 0, int(255 // max_level * (level)))
        for image_path in image_path_list:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                trans_mask = image[:, :, 3] == 0
                image[trans_mask] = [255, 255, 255, 255]
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            resize_padding_image = resize_padding(image, width + 5, height + 5)
            write_char_image = write_char(resize_padding_image, level, color)
            transparent_image = transparent(write_char_image, level, color)
            image_dict[level].append(transparent_image)

    thumbnail_image = thumbnail(image_dict, col_num)
    cv2.imwrite('result.png', thumbnail_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='portfolio')
    parser.add_argument('--image_dir_path', type=str, default=os.path.expanduser('~/Desktop/image'))
    parser.add_argument('--col_num', type=int, default=7)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--height', type=int, default=128)
    args = parser.parse_args()
    main(args.image_dir_path, args.col_num, args.width, args.height)