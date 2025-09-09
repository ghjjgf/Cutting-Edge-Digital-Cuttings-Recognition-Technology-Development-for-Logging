import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示汉字
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def predict_image_color(model, image_path, n_clusters=8):
    """
    使用训练好的模型预测新图片的颜色，并返回前5大颜色及其占比。
    """
    print(f"\n--- 正在预测图片: {os.path.basename(image_path)} ---")
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        reshaped_img = img_array.reshape(-1, 3)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init='auto',
            batch_size=4096
        )
        kmeans.fit(reshaped_img)
        cluster_centers = kmeans.cluster_centers_
        predictions = model.predict(cluster_centers)
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        print(f"unique_predictions: {unique_predictions}, counts: {counts}")
        most_common_prediction = unique_predictions[np.argmax(counts)]
        labels = kmeans.labels_
        counts_pixels = np.bincount(labels)
        top_5_indices = np.argsort(counts_pixels)[::-1][:5]
        top_5_colors = []
        total_pixels = len(reshaped_img)
        for index in top_5_indices:
            color_rgb = tuple(cluster_centers[index].astype(int))
            percentage = (counts_pixels[index] / total_pixels) * 100
            top_5_colors.append({'rgb': color_rgb, 'percentage': percentage})
        return most_common_prediction, top_5_colors
    except Exception as e:
        print(f"处理图片 '{image_path}' 时出错: {e}")
        return "无法识别", None


if __name__ == '__main__':
    model_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/project/model/color_classifier_knn.joblib"
    if os.path.exists(model_path):
        print("--- 正在加载已存在的模型 ---")
        knn_model = joblib.load(model_path)
        print("模型加载成功！")
    else:
        print(f"错误：模型文件 '{model_path}' 不存在。请先训练模型。")
        knn_model = None

    # 预测并显示结果
    if knn_model:
        test_image_path = "D:/计算机视觉学习/deep-learning-for-image-processing-master/data_set/cuttings/train/Dark-gray-sandy-mudstone/Dark-gray-sandy-mudstone_20.webp"
        if os.path.exists(test_image_path):
            predicted_color, top_5_colors = predict_image_color(knn_model, test_image_path)
            if top_5_colors:
                print(f"该图片的主要颜色是 '{predicted_color}'")
                print("图片中占比最高的5种颜色及其占比：")
                for i, color_data in enumerate(top_5_colors):
                    print(f"{i + 1}. RGB: {color_data['rgb']}, 占比: {color_data['percentage']:.2f}%")
            else:
                print("无法获取颜色数据。")
        else:
            print(f"错误：测试图片 '{test_image_path}' 不存在。")
