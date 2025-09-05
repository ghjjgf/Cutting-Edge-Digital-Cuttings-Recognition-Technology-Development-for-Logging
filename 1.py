import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import os

COLOR_NAMES = {
    '灰白色': (104,85, 70),
    '灰绿色': (78, 74, 68),
    '灰色': (44, 40, 54),
    '浅灰色': (37, 33, 30),
    '深灰色': (48, 48, 53),
    '紫红色': (65, 52, 60)
}


def get_color_name(rgb_tuple):
    """
    根据给定的RGB值，返回最接近的颜色名称。
    通过遍历预定义的精简颜色字典并计算欧氏距离来找到最匹配的颜色。

    Args:
        rgb_tuple (tuple): 一个 (R, G, B) 颜色元组。

    Returns:
        str: 最接近的颜色名称（可能带有“最接近”的提示）。
    """
    min_distance = float('inf')
    closest_color_name = '未知颜色'
    for color_name, color_rgb in COLOR_NAMES.items():
        distance = np.linalg.norm(np.array(rgb_tuple) - np.array(color_rgb))
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name
    if min_distance < 10:
        return closest_color_name
    elif min_distance < 50:
        return f"{closest_color_name}（最接近）"
    else:
        return f"未知颜色（最接近 {closest_color_name}）"


def get_most_common_color_kmeans(image_path, n_clusters=4):
    """
    使用K-means算法提取图片中最主要（占比最高）的颜色，
    并返回其RGB值和占比。

    Args:
        image_path (str): 图片文件的路径。
        n_clusters (int): 聚类的数量，即你想提取的“主色”数量。

    Returns:
        tuple: 包含最主要颜色（RGB元组）和其占比（百分比）的元组。
               如果无法处理图片，则返回 (None, None)。
    """
    batch_size = 4096
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        reshaped_img = img_array.reshape(-1, 3)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init='auto',
            batch_size=batch_size
        )
        kmeans.fit(reshaped_img)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        most_common_cluster_index = np.argmax(counts)
        most_common_count = counts[most_common_cluster_index]
        most_common_color = kmeans.cluster_centers_[most_common_cluster_index]
        most_common_color_rgb = tuple(most_common_color.astype(int))
        total_pixels = len(reshaped_img)
        percentage = (most_common_count / total_pixels) * 100
        return most_common_color_rgb, percentage
    except FileNotFoundError:
        print(f"错误：文件路径 '{image_path}' 不存在。跳过此文件。")
        return None, None
    except Exception as e:
        print(f"处理文件 '{os.path.basename(image_path)}' 时发生错误：{e}")
        return None, None


def process_images_in_folder(folder_path):
    """
    遍历指定文件夹中的所有图片，并处理每一张图片。

    Args:
        folder_path (str): 包含图片的文件夹路径。
    """
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹路径 '{folder_path}' 不存在。")
        return
    print(f"开始处理文件夹：{folder_path}\n")
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('小视野.webp'):
            image_path = os.path.join(folder_path, filename)
            print(f"--- 正在处理图片：{filename} ---")
            dominant_color, percentage = get_most_common_color_kmeans(image_path, n_clusters=8)
            if dominant_color:
                color_name = get_color_name(dominant_color)
                print(f"图片中最主要（占比最高）的颜色为：{color_name}")
                print(f"该颜色的RGB值为：{dominant_color}")
                print(f"该颜色在整个图片中的占比：{percentage:.2f}%\n")
            else:
                print("无法处理此图片。\n")


folder_path_to_process = 'dataset/train/灰色泥质粉砂岩'
process_images_in_folder(folder_path_to_process)
