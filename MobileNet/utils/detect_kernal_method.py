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

def process_image_blockwise(image_path, rgb_range, h_range, save_path, block_size=3, stride=1, padding=2):
    """
    使用检测块滑动遍历图像，若检测块内所有像素都在范围内，则绘制红色，否则绘制蓝色
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape

    # 转HSV
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    img_h = img_hsv[:, :, 0] * 2  # 转换为 0-360° 范围

    # padding
    img_np_padded = np.pad(img_np, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    img_h_padded = np.pad(img_h, ((padding, padding), (padding, padding)), mode='edge')

    out_img = np.zeros_like(img_np)

    half = block_size // 2
    for y in tqdm(range(padding, h + padding), desc=f"处理 {os.path.basename(image_path)}", ncols=80):
        for x in range(padding, w + padding, stride):
            # 获取检测块
            block_rgb = img_np_padded[y-half:y+half+1, x-half:x+half+1, :]
            block_h = img_h_padded[y-half:y+half+1, x-half:x+half+1]

            # 判断检测块是否全部在范围内
            in_range = True
            for by in range(block_size):
                for bx in range(block_size):
                    r, g, b = block_rgb[by, bx]
                    h_val = block_h[by, bx]
                    if not pixel_in_range(r, g, b, h_val, rgb_range, h_range):
                        in_range = False
                        break
                if not in_range:
                    break

            # 将检测块绘制到输出图像
            color = [255, 0, 0] if in_range else [0, 0, 255]
            out_img[y-padding-half:y-padding+half+1, x-padding-half:x-padding+half+1] = color

    # 保存结果
    Image.fromarray(out_img).save(save_path)
    print(f"已保存: {save_path}")

# ==================== 示例用法 ====================
if __name__ == "__main__":
    image_path = "H:/dataset/紫红色泥岩/A/Purple-red-mudstone_36.webp"
    save_path = "C:/Users/28162/Desktop/webp_filled/kernal_purple_red22.webp"
    rgb_range = [(60, 100), (20, 65), (10, 55)]  # R,G,B范围
    h_range = (3, 20)  # H范围
    process_image_blockwise(image_path, rgb_range, h_range, save_path)
