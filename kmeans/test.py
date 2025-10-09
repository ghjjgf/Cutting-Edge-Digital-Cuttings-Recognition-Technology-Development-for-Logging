import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2hsv
from tqdm import tqdm  # 导入tqdm库

# 计算邻域内的RGB和HSV特征
def extract_features(image, x, y, window_size=3):
    half_window = window_size // 2
    patch = image[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
    
    # 获取RGB特征：3x3的像素点的27维
    rgb_values = patch.reshape(-1, 3)  # 9个像素的RGB特征，3通道每个像素
    
    # 获取HSV特征：9个像素的H值
    hsv_patch = rgb2hsv(patch / 255.0)  # 将RGB归一化到[0, 1]范围
    h_values = hsv_patch[:, :, 0].reshape(-1)  # 提取H通道
    
    # 计算RGB均值：9个像素的RGB均值
    rgb_mean = np.mean(rgb_values, axis=0)  # 9个像素的RGB均值
    
    # 计算H均值：9个像素的H均值
    h_mean = np.mean(h_values)  # 9个像素的H均值
    
    # 返回38维特征：27维RGB + 9维H + 1维RGB均值 + 1维H均值
    feature = np.concatenate([rgb_values.flatten(), h_values, rgb_mean, [h_mean]])
    return feature

# 加载图像并转换为RGB格式
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 提取图像的特征
def extract_image_features(image, window_size=3):
    height, width, _ = image.shape
    features = []
    
    # 使用tqdm显示进度条
    for y in tqdm(range(window_size//2, height - window_size//2), desc="Extracting features", unit="row"):
        for x in range(window_size//2, width - window_size//2):
            feature = extract_features(image, x, y, window_size)
            features.append(feature)
    
    return np.array(features)

# 执行KMeans聚类
def kmeans_clustering(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans

# 将聚簇中心映射为颜色
def define_cluster_centers(kmeans):
    # 红、黄、蓝的RGB表示
    red = np.array([255, 0, 0])
    yellow = np.array([255, 255, 0])
    blue = np.array([0, 0, 255])
    
    # 获取聚簇中心对应的颜色
    centers = np.array([red, yellow, blue])
    
    return centers

# 将每个像素分类为红、黄、蓝并上色
def classify_pixels(image, kmeans, cluster_centers, window_size=3):
    height, width, _ = image.shape
    classified_image = np.zeros((height, width, 3), dtype=np.uint8)  # 结果为RGB图像
    
    # 使用tqdm显示进度条
    for y in tqdm(range(window_size//2, height - window_size//2), desc="Classifying pixels", unit="row"):
        for x in range(window_size//2, width - window_size//2):
            feature = extract_features(image, x, y, window_size)
            feature = feature.reshape(1, -1)  # 转为行向量
            
            # 预测所属类别
            label = kmeans.predict(feature)
            
            # 将分类结果映射到红、黄、蓝
            classified_image[y, x] = cluster_centers[label]
    
    return classified_image

# 主流程
def main(image_path, output_path):
    image = load_image(image_path)
    features = extract_image_features(image)
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 执行KMeans聚类
    kmeans = kmeans_clustering(features_scaled)
    
    # 获取聚簇中心的颜色映射
    cluster_centers = define_cluster_centers(kmeans)
    
    # 分类像素并上色
    classified_image = classify_pixels(image, kmeans, cluster_centers)
    
    # 保存上色后的图像
    cv2.imwrite(output_path, classified_image)  # 保存为指定路径
    
    # 显示分类结果
    cv2.imshow('Classified Image', classified_image)  # 显示上色后的图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 调用主函数
if __name__ == "__main__":
    image_path = "C:\\Users\\28162\\Desktop\\Purple-red-mudstone_37.webp"  # 替换为你自己的图像路径
    output_path = "C:\\Users\\28162\\Desktop\\classified_image.jpg"  # 指定保存图像的路径
    main(image_path, output_path)
