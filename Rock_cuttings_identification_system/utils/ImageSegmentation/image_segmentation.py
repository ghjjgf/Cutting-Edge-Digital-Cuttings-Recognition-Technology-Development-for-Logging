import numpy as np
import cv2
from PIL import Image

class ImageSegmentation():
    def __init__(self):
        """初始化图像分割器"""
        pass
    
    import numpy as np


    def segment_by_clusters(self, image, cluster_labels, n_clusters=3):
        """
        根据聚类标签将图像分割为多个子图像
        Args:
            image: 原始图像 (numpy array 或 文件路径)
            cluster_labels: 聚类标签 (numpy array, 与image同尺寸)
            n_clusters: 聚类数量
        Returns:
            segmented_images: 分割后的子图像列表 (numpy array list)
        """
        # 如果传入是路径 -> 读取
        if isinstance(image, str):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))

        h, w, c = image.shape
        segmented_images = []
        
        for cluster_id in range(n_clusters):
            # 创建掩码
            mask = (cluster_labels == cluster_id).astype(np.uint8)

            # 应用掩码
            segmented_image = image.copy()
            segmented_image[mask == 0] = [0, 0, 0]

            segmented_images.append(segmented_image)
        
        return segmented_images

    
    def calculate_cluster_ratio(self, cluster_labels, n_clusters=3):
        """
        计算每个聚类的像素占比
        Args:
            cluster_labels: 聚类标签 (numpy array)
            n_clusters: 聚类数量
        Returns:
            ratios: 每个聚类的像素占比列表
        """
        total_pixels = cluster_labels.size
        ratios = []
        
        for cluster_id in range(n_clusters):
            cluster_pixels = np.sum(cluster_labels == cluster_id)
            ratio = cluster_pixels / total_pixels
            ratios.append(ratio)
        
        return ratios
    
    def filter_empty_segments(self, segmented_images, min_pixel_count=100):
        """
        过滤掉像素数量过少的子图像
        Args:
            segmented_images: 分割后的子图像列表
            min_pixel_count: 最小像素数量阈值
        Returns:
            filtered_images: 过滤后的子图像列表
            valid_indices: 有效子图像的索引列表
        """
        filtered_images = []
        valid_indices = []
        
        for i, img in enumerate(segmented_images):
            # 计算非黑色像素数量
            non_black_pixels = np.sum(np.any(img != [0, 0, 0], axis=2))
            
            if non_black_pixels >= min_pixel_count:
                filtered_images.append(img)
                valid_indices.append(i)
        
        return filtered_images, valid_indices
