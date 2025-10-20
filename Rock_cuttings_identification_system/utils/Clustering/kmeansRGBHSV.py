import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

class clusterKMeans:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, image):
        """
        对输入图像进行KMeans聚类，使用每像素的 RGB + HSV 六维特征
        Args:
            image: 输入图像路径（支持 PIL 打开）
        Returns:
            cluster_centers: 聚类中心（shape: n_clusters x 6）
            labels: 每个像素的聚类标签（H x W）
        """
        # 使用 PIL 保证兼容各种格式并得到 RGB 顺序
        img = Image.open(image).convert('RGB')
        img_array = np.array(img)  # H x W x 3 (RGB)

        h, w = img_array.shape[:2]

        # 计算 HSV（OpenCV 使用 BGR 输入），先将 RGB -> BGR
        bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)  # H:0-179, S:0-255, V:0-255

        # 可选：将 H 扩展到 0-255 范围，使 RGB 与 H 在同一数量级（便于聚类）
        hsv_scaled = hsv.copy()
        hsv_scaled[..., 0] = (hsv[..., 0].astype(np.float32) * (255.0 / 179.0)).astype(np.uint8)

        # 展平并拼接 RGB 和 HSV（已缩放 H），得到 N x 6 特征
        rgb_flat = img_array.reshape(-1, 3).astype(np.float32)
        hsv_flat = hsv_scaled.reshape(-1, 3).astype(np.float32)
        features = np.concatenate([rgb_flat, hsv_flat], axis=1)  # N x 6

        # 使用 MiniBatchKMeans 进行聚类
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=0,
            n_init='auto',
            batch_size=4096
        )
        kmeans.fit(features)

        # 保存聚类中心与标签（标签恢复为 H x W）
        self.cluster_centers_ = kmeans.cluster_centers_  # n_clusters x 6
        self.labels_ = kmeans.labels_.reshape(h, w)

        return self.cluster_centers_, self.labels_

    def get_cluster_centers(self):
        """获取聚类中心（n_clusters x 6）"""
        return self.cluster_centers_

    def get_labels(self):
        """获取聚类标签（H x W）"""
        return self.labels_