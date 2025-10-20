# import numpy as np
# import cv2
# from sklearn.cluster import KMeans
# from PIL import Image
# from sklearn.cluster import MiniBatchKMeans

# class clusterKMeans:
#     def __init__(self, n_clusters=3):
#         self.n_clusters = n_clusters
#         self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         self.cluster_centers_ = None
#         self.labels_ = None

#     def fit(self, image):
#         """
#         对输入图像进行KMeans聚类
#         Args:
#             image: 输入图像 (numpy array)
#         Returns:
#             cluster_centers: 聚类中心
#             labels: 每个像素的聚类标签
#         """
#         # 将图像重塑为像素点数据
#         img = Image.open(image).convert('RGB')
#         img_array = np.array(img)
#         reshaped_img = img_array.reshape(-1, 3)
#         kmeans = MiniBatchKMeans(
#             n_clusters=self.n_clusters,
#             random_state=0,
#             n_init='auto',
#             batch_size=4096
#         )
#         kmeans.fit(reshaped_img)
#         # 进行KMeans聚类
#         self.cluster_centers_ = kmeans.cluster_centers_
#         self.labels_ = kmeans.labels_.reshape(img_array.shape[0], img_array.shape[1])
#         return self.cluster_centers_, self.labels_

#     def get_cluster_centers(self):
#         """获取聚类中心"""
#         return self.cluster_centers_

#     def get_labels(self):
#         """获取聚类标签"""
#         return self.labels_