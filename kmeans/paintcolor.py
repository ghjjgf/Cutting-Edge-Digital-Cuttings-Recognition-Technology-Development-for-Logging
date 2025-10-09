import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm

# 提取特征并计算RGB均值、H均值
def extract_features(image, x, y, window_size=3):
    half_window = window_size // 2
    patch = image[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
    
    # 提取RGB特征：3x3像素
    rgb_values = patch.reshape(-1, 3)  # 9个像素点的RGB特征
    
    # 提取H特征：9个像素的H值
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    h_values = (patch_hsv[:, :, 0] * 2).reshape(-1)  # H通道转为0-360°

    # 计算RGB均值和H均值
    rgb_mean = np.mean(rgb_values, axis=0)
    h_mean = np.mean(h_values)
    
    # 返回40维特征：27维RGB + 9维H + 3维RGB均值 + 1维H均值
    return np.concatenate((rgb_values.flatten(), h_values, rgb_mean, [h_mean]))

def set_rgbh_list():
    rgbh_list1 = {
        'rgb': [
            [95.29751309, 69.45374446, 57.28960646],
            [95.4179513, 69.32362202, 56.81548566],
            [95.3531213, 68.98606774, 56.46741104],
            [95.28623434, 69.73938324, 57.80939544],
            [95.4446978, 69.6555729, 57.24024994],
            [95.41883772, 69.32811633, 56.84629791],
            [95.11790556, 69.80488331, 58.42492893],
            [95.29033279, 69.75183629, 57.84443136],
            [95.30377409, 69.4694561, 57.35093101]
        ],
        'h': [20.86977136, 19.20793331, 20.79109746, 20.82327144, 17.49333353, 19.25999504,  
            24.2197898, 20.86702889, 20.97423482],
        'rgb_mean': [95.32559644, 69.50140915, 57.34319308],
        'h_mean': 20.5007172954488
    }
    
    rgbh_list2 = {
        'rgb': [
            [59.52255251, 47.29572797, 50.46319004],
            [60.31337946, 48.33783042, 51.71451069],
            [61.24210777, 49.64262054, 51.98108916],
            [58.94345139, 46.22987407, 49.83930688],
            [59.54152406, 47.06901767, 51.36566166],
            [60.24842245, 48.30507005, 51.68680775],
            [58.44712327, 45.48587892, 48.31298111],
            [58.87854313, 46.17722493, 49.77858039],
            [59.39618446, 47.21748576, 50.38094011]
        ],
        'h': [106.48469045, 110.47663613, 99.80007602, 111.30288899, 121.7266798,
            110.47429449, 101.52447877, 111.34629236, 106.46093661],
        'rgb_mean': [59.61480983, 47.30674781, 50.6136742],
        'h_mean': 108.84410817986803
    }
    
    rgbh_list3 = {
        'rgb': [
            [56.77829183, 34.58379607, 25.75217286],
            [56.50286423, 34.34404562, 25.60047009],
            [56.26785261, 34.11864679, 25.674418  ],
            [56.94976447, 34.76394198, 25.70588529],
            [56.71278783, 34.56198669, 25.51680352],
            [56.52408017, 34.35178455, 25.59492366],
            [57.16502883, 34.94705378, 25.87487388],
            [56.9694053, 34.77389135, 25.70791878],
            [56.81760755, 34.59997781, 25.74884435]
        ],
        'h': [15.85955997, 15.43725014, 17.79037492, 14.50850903, 12.97444866, 15.41245917,
              15.826553, 14.47326793, 15.81423197],
        'rgb_mean': [56.74307587, 34.5605694, 25.68625671],
        'h_mean': 15.344072752555512
    }
    
    return [rgbh_list1, rgbh_list2, rgbh_list3]

def get_nearest_cluster(features, clusters):
    # 计算该像素点与各聚簇中心的欧几里得距离
    cluster_centers = np.array([
        np.concatenate([np.array(cluster['rgb']).flatten(), np.array(cluster['h']), np.array(cluster['rgb_mean']), np.array([cluster['h_mean']])])
        for cluster in clusters
    ])
    
    distances = cdist([features], cluster_centers, metric='euclidean')
    
    # 返回最小距离对应的聚簇索引
    return np.argmin(distances)

def main(image_path):
    # 获取预设的RGBH列表
    clusters = set_rgbh_list()
    
    # 读取并转换图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape

    # 创建输出图像
    output_img = img.copy()

    # 设置颜色：红色、黄色、蓝色
    colors = [[255, 0, 0], [255, 255, 0], [0, 0, 255]]
    
    # 遍历图像中的每个像素，添加进度条
    for y in tqdm(range(1, height-1), desc="处理进度", unit="行"):
        for x in range(1, width-1):
            features = extract_features(img, x, y)
            
            # 获取与聚簇中心最接近的聚簇
            cluster_idx = get_nearest_cluster(features, clusters)
            
            # 使用对应颜色标记该像素
            output_img[y, x] = colors[cluster_idx]
    
    # 保存修改后的图像
    output_path = "output_image22.png"
    Image.fromarray(output_img).save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == '__main__':
    image_path = "C:\\Users\\28162\\Desktop\\Purple-red-mudstone_37.webp"  # 修改为你的图像路径
    main(image_path)
