import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.color import rgb2hsv
from tqdm import tqdm
import os

# 计算邻域内的RGB和HSV特征
def extract_features(image, x, y, window_size=3):
    half_window = window_size // 2
    patch = image[y-half_window:y+half_window+1, x-half_window:x+half_window+1]

    rgb_values = patch.reshape(-1, 3)
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    patch_hsv = patch_hsv[:, :, 0].flatten()  # H通道

    rgb_mean = np.mean(rgb_values, axis=0)
    h_mean = np.mean(patch_hsv)

    feature = np.concatenate([rgb_values.flatten(), patch_hsv, rgb_mean, [h_mean]])
    return feature

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_all_features(image, window_size=3):
    height, width, _ = image.shape
    features = []
    coords = []
    for y in tqdm(range(window_size//2, height - window_size//2), desc="Extracting features", unit="row"):
        for x in range(window_size//2, width - window_size//2):
            feature = extract_features(image, x, y, window_size)
            features.append(feature)
            coords.append((y, x))
    return np.array(features), coords

def kmeans_clustering(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans

def colorize_image(image, features, coords, cluster_centers):
    from scipy.spatial.distance import cdist
    distances = cdist(features, cluster_centers, metric='euclidean')
    labels = np.argmin(distances, axis=1)

    cluster_colors = [
        np.array([255, 0, 0]),   # 红
        np.array([255, 255, 0]), # 黄
        np.array([0, 0, 255])    # 蓝
    ]

    result_img = np.zeros_like(image)
    for (y, x), label in zip(coords, labels):
        result_img[y, x] = cluster_colors[label]
    return result_img

def main(image_path, save_path):
    image = load_image(image_path)
    all_features, coords = extract_all_features(image)
    kmeans = kmeans_clustering(all_features, n_clusters=3)

    labels = kmeans.labels_
    counts_pixels = np.bincount(labels)
    total_pixels = len(all_features)
    for i in range(3):
        print(f"聚簇 {i+1} 占比: {counts_pixels[i] / total_pixels * 100:.2f}%")

    result_img = colorize_image(image, all_features, coords, kmeans.cluster_centers_)

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"结果已保存至: {save_path}")

if __name__ == "__main__":
    image_path = "C:/Users/28162/Desktop/dataset/gray_green_cuttings/2325_gray_green_cuttings.webp"
    save_path = "C:/Users/28162/Desktop/中石油课题/val_img/kmeans_knn_val/灰绿色泥岩"  # 这里改成你要保存的路径
    main(image_path, save_path)
