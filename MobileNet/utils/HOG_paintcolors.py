import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def pixel_in_range(r, g, b, h, rgb_range, h_range):
    """判断单个像素的RGB和H是否在给定的范围内"""
    r_min, r_max = rgb_range[0]
    g_min, g_max = rgb_range[1]
    b_min, b_max = rgb_range[2]
    h_min, h_max = h_range
    return (r_min <= r <= r_max) and (g_min <= g <= g_max) and (b_min <= b <= b_max) and (h_min <= h <= h_max)

def compute_hog_pixel(img_gray, bins=6):
    """计算整张灰度图每个像素的HOG向量，返回 h x w x bins"""
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle[angle < 0] += 360

    h_img, w_img = img_gray.shape
    hog_map = np.zeros((h_img, w_img, bins), dtype=np.float32)

    bin_edges = np.linspace(0, 360, bins + 1)
    bin_indices = np.digitize(angle, bin_edges) - 1
    bin_indices[bin_indices == bins] = bins - 1

    for b in range(bins):
        hog_map[:, :, b] = np.where(bin_indices == b, mag, 0)

    # 归一化
    hog_sum = np.sum(hog_map, axis=2, keepdims=True) + 1e-6
    hog_map /= hog_sum
    return hog_map

def pixel_hog_threshold(hog_vector, hog_min, hog_max, count_threshold=3):
    """
    判断HOG向量中有多少个维度在阈值范围内
    如果大于count_threshold个维度满足条件，返回True
    """
    mask = (hog_vector >= hog_min) & (hog_vector <= hog_max)
    return np.sum(mask) > count_threshold

def process_image_pixelwise_hog(image_path, rgb_range, h_range, hog_min, hog_max, save_path):
    """
    遍历每个像素，同时判断RGB/H和HOG六维阈值
    同时满足条件标红，否则标蓝
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h_img, w_img, _ = img_np.shape

    # HSV
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    img_h = img_hsv[:, :, 0] * 2   # 转为 0-360°

    # 灰度图
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    hog_map = compute_hog_pixel(img_gray, bins=len(hog_min))

    out_img = np.zeros_like(img_np)

    for y in tqdm(range(h_img), desc=f"处理 {os.path.basename(image_path)}", ncols=80):
        for x in range(w_img):
            r, g, b = img_np[y, x]
            h_val = img_h[y, x]
            hog_vec = hog_map[y, x]

            # 判断RGB/H和HOG条件
            rgb_h_ok = pixel_in_range(r, g, b, h_val, rgb_range, h_range)
            hog_ok = pixel_hog_threshold(hog_vec, hog_min, hog_max, count_threshold=3)

            if rgb_h_ok and hog_ok:
                out_img[y, x] = [255, 0, 0]  # 红色
            else:
                out_img[y, x] = [0, 0, 255]  # 蓝色

    Image.fromarray(out_img).save(save_path)
    print(f"✅ 已保存: {save_path}")

# ==================== 示例用法 ====================
if __name__ == "__main__":
    image_path = "H:/dataset/紫红色泥岩/A/Purple-red-mudstone_37.webp"
    save_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/webp_filled/紫红色泥岩/hog.webp"

    cutting_class = {
        0 : "深灰色泥岩",
        1 : "深灰色粉砂质泥岩",
        2 : "灰绿色泥岩",
        3 : "灰色泥质粉砂岩",
        4 : "浅灰色中砂岩",
        5 : "灰白色中砂岩",
        6 : "紫红色泥岩"
    } 
    cutting_classNum = 6  # 选择不同类别

    if cutting_classNum == 0:
        rgb_range = [(20, 60), (20, 60), (20, 60)]
        h_range = (80, 100)
        hog_min = np.array([0, 0, 0, 0, 0, 0])
        hog_max = np.array([1, 1, 1, 1, 1, 1])

    elif cutting_classNum == 1:
        rgb_range = [(20, 60), (20, 60), (20, 60)]
        h_range = (70, 90)
        hog_min = np.array([0, 0, 0, 0, 0, 0])
        hog_max = np.array([1, 1, 1, 1, 1, 1])

    elif cutting_classNum == 2:
        rgb_range = [(55, 95), (60, 90), (55, 90)]
        h_range = (50, 75)
        hog_min = np.array([0, 0, 0, 0, 0, 0])
        hog_max = np.array([1, 1, 1, 1, 1, 1])  

    elif cutting_classNum == 3:
        rgb_range = [(50, 90), (51, 91), (50, 90)]
        h_range = (60, 90)
        hog_min = np.array([0, 0, 0, 0, 0, 0])
        hog_max = np.array([1, 1, 1, 1, 1, 1])

    elif cutting_classNum == 4:
        rgb_range = [(50, 90), (51, 91), (50, 90)]
        h_range = (60, 90)
        hog_min = np.array([0, 0, 0, 0, 0, 0])
        hog_max = np.array([1, 1, 1, 1, 1, 1])

    elif cutting_classNum == 5:
        rgb_range = [(70, 110), (68, 110), (70, 110)]
        h_range = (77, 97)
        hog_min = np.array([0, 0, 0, 0, 0, 0])
        hog_max = np.array([1, 1, 1, 1, 1, 1])

    elif cutting_classNum == 6:
        rgb_range = [(60, 100), (20, 65), (10, 55)]
        h_range = (3, 20)
        hog_min = np.array([0, 0, 0, 0, 0, 0])
        hog_max = np.array([1, 1, 1, 1, 1, 1])

    process_image_pixelwise_hog(image_path, rgb_range, h_range, hog_min, hog_max, save_path)
