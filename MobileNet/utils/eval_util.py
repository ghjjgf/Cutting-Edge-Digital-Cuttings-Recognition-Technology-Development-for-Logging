import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def pixel_in_range(r, g, b, h, rgb_range, h_range):
    """
    判断单个像素的RGB和H是否都在给定的范围内
    """
    r_min, r_max = rgb_range[0]
    g_min, g_max = rgb_range[1]
    b_min, b_max = rgb_range[2]
    h_min, h_max = h_range

    return (r_min <= r <= r_max) and (g_min <= g <= g_max) and (b_min <= b <= b_max) and (h_min <= h <= h_max)

def process_image_pixelwise(image_path, rgb_range, h_range, save_path):
    """
    对单张图片遍历所有像素，判断是否在范围内，落在范围内为红色，否则为蓝色
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape

    # 转HSV
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    img_h = img_hsv[:,:,0]

    # 输出图像初始化
    out_img = np.zeros_like(img_np)

    for y in tqdm(range(h), desc=f"处理 {os.path.basename(image_path)}", ncols=80):
        for x in range(w):
            r, g, b = img_np[y, x]
            h_val = img_h[y, x]
            if pixel_in_range(r, g, b, h_val, rgb_range, h_range):
                out_img[y, x] = [255, 0, 0]  # 红色
            else:
                out_img[y, x] = [0, 0, 255]  # 蓝色

    # 保存结果
    Image.fromarray(out_img).save(save_path)
    print(f"已保存: {save_path}")

# 示例用法
if __name__ == "__main__":
    image_path = "H:/dataset/紫红色泥岩/A/Purple-red-mudstone_36.webp"
    save_path = "C:/Users/28162/Desktop/webp_filled/Purple-red-mudstone_36_filled.webp"
    # 设定RGB和H范围
    rgb_range = [(60, 100), (20, 65), (10, 55)]  # R,G,B范围
    h_range = (3, 20)  # H范围
    process_image_pixelwise(image_path, rgb_range, h_range, save_path)