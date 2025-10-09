import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2hsv
from tqdm import tqdm  # 导入tqdm库

# 计算邻域内的RGB和HSV特征
def extract_features(image, x, y, window_size=3):
    half_window = window_size // 2
    # 提取3x3邻域
    patch = image[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
    
    # 获取RGB特征：3x3的像素点的27维
    rgb_values = patch.reshape(-1, 3)  # 9个像素的RGB特征，3通道每个像素

    # 获取HSV特征：9个像素的H值（0-360°）
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    patch_hsv = patch_hsv[:,:,0].flatten()  # H通道，转换为0-360°范围
    #print(patch_hsv)
    
    # 计算RGB均值：9个像素的RGB均值
    rgb_mean = np.mean(rgb_values, axis=0)  # 9个像素的RGB均值

    # 计算H均值：9个像素的H均值
    h_mean = np.mean(patch_hsv)  # 9个像素的H均值

    # 返回40维特征：27维RGB + 9维H + 3维RGB均值 + 1维H均值
    feature = np.concatenate([rgb_values.flatten(), patch_hsv, rgb_mean, [h_mean]])
    return feature

# 加载图像并转换为RGB格式
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 提取图像的特征（增量处理，避免一次性加载所有特征）
def extract_image_features(image, window_size=3):
    height, width, _ = image.shape
    features = []
    
    # 使用tqdm显示进度条
    for y in tqdm(range(window_size//2, height - window_size//2), desc="Extracting features", unit="row"):
        for x in range(window_size//2, width - window_size//2):
            feature = extract_features(image, x, y, window_size)
            features.append(feature)
            
            # 分批处理特征，避免内存溢出
            if len(features) >= 10000:  # 每次处理10000个特征
                yield np.array(features)
                features = []
    
    # 处理剩余的特征
    if len(features) > 0:
        yield np.array(features)

# 执行KMeans聚类
def kmeans_clustering(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans

# 反标准化KMeans聚类中心
def unscale_centers(centers, scaler):
    # 使用Scaler的参数将聚类中心反标准化
    return scaler.inverse_transform(centers)

# 主流程
def main(image_path):
    image = load_image(image_path)
    
    # 使用增量方式提取特征，避免内存溢出
    all_features = []
    for features_batch in extract_image_features(image):
        all_features.append(features_batch)
    
    # 合并所有批次的特征
    all_features = np.concatenate(all_features, axis=0)
    
    # 标准化特征
    #scaler = StandardScaler()
    #all_features = scaler.fit_transform(all_features)
    
    # 执行KMeans聚类
    kmeans = kmeans_clustering(all_features)
    #print(f"聚簇中心:\n{kmeans.cluster_centers_}")
    labels = kmeans.labels_
    counts_pixels = np.bincount(labels)
    top_5_indices = np.argsort(counts_pixels)[::-1][:3]
    top_3_colors = []
    total_pixels = len(all_features)
    for index in top_5_indices:
        percentage = (counts_pixels[index] / total_pixels) * 100
        top_3_colors.append({
            'percentage': percentage
        })
    print(f"前三类颜色占比: {top_3_colors}")

    # 反标准化聚簇中心
    # original_centers = unscale_centers(kmeans.cluster_centers_, scaler)
    # print(f"反标准化后的聚簇中心:{original_centers}")
    
    # 输出反标准化后的RGB和H值
    print("三类聚簇中心（原始RGB和H值）:")
    for i, center in enumerate(kmeans.cluster_centers_):
        # 将每个聚簇的中心拆分为RGB和H值
        rgb = center[:27].reshape(9, 3)  # 前27个是RGB特征
        h_values = center[27:36]  # 接下来9个是H特征
        print(h_values)
        rgb_mean = center[36:39]  # 3个是RGB均值
        h_mean = center[39]  # 1个是H均值
        top_3_colors.append({
            '聚簇' : i + 1,
            'rgb': rgb,
            'h_values': h_values,
            'rgb_mean': rgb_mean,
            'h_mean': h_mean
        })
        print(top_3_colors)
        # print(f"聚簇 {i + 1}:")
        # print(f"  RGB特征中心 (9个像素): {rgb}")
        # print(f"  H值特征中心: {h_values}")  # 将H值转换为[0, 360]的范围
        # print(f"  RGB均值: {rgb_mean}")
        # print(f"  H均值: {h_mean}")  # 将H均值转换为[0, 360]的范围
        
# 调用主函数
if __name__ == "__main__":
    image_path = "C:\\Users\\28162\\Desktop\\Purple-red-mudstone_37.webp"  # 替换为你自己的图像路径
    main(image_path)
