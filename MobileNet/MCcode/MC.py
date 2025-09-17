import cv2
import numpy as np
import os
import random
from PIL import Image

def read_image(image_path):
    """兼容读取 jpg/png/webp 格式"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        # 用 Pillow 兜底
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise FileNotFoundError(f"无法读取图像: {image_path}, 错误信息: {e}")
    return img

def calculate_iou(rect1, rect2):
    """计算两个矩形框的IoU"""
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    return intersection_area / float(area1 + area2 - intersection_area)

def monte_carlo_crop(image_path, save_dir, num_crops=8, min_size=(640, 640), max_iou=0.25, save_as_webp=True):
    """进行随机裁剪并确保每两个框之间的IoU小于 max_iou"""
    img = read_image(image_path)
    h, w, _ = img.shape
    print(f"输入图像大小: {w}x{h}")

    os.makedirs(save_dir, exist_ok=True)
    rects = []

    for i in range(num_crops):
        while True:
            crop_w = random.randint(min_size[0], w // 2)
            crop_h = random.randint(min_size[1], h // 2)
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            x2, y2 = x1 + crop_w, y1 + crop_h
            new_rect = (x1, y1, x2, y2)

            if all(calculate_iou(rect, new_rect) < max_iou for rect in rects):
                rects.append(new_rect)
                crop_img = img[y1:y2, x1:x2]

                # 用 Pillow 保存，确保 WebP/JPG 都可以
                crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                ext = "webp" if save_as_webp else "jpg"
                file_name = os.path.basename(image_path)
                save_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}_crop_{i+1}.{ext}")
                try:
                    crop_pil.save(save_path, format=ext.upper())
                    print(f"已保存: {save_path}, 尺寸: {crop_w}x{crop_h}")
                except Exception as e:
                    print(f"保存图像时发生错误: {e}")
                    continue
                break

    print(f"{os.path.basename(image_path)} 所有裁剪完成！")

def crop_folder_images(input_folder, output_folder, num_crops=8, min_size=(640, 640), max_iou=0.25, save_as_webp=True):
    """对文件夹内所有图片进行裁剪"""
    exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]
    for f in files:
        image_path = os.path.join(input_folder, f)
        monte_carlo_crop(
            image_path, 
            output_folder, 
            num_crops=num_crops, 
            min_size=min_size, 
            max_iou=max_iou, 
            save_as_webp=save_as_webp
        )

# 使用示例
if __name__ == "__main__":
    input_folder = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/MCcode/your_input_folder"
    output_folder = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/MCcode/MC_crop"
    crop_folder_images(input_folder, output_folder, save_as_webp=True)
