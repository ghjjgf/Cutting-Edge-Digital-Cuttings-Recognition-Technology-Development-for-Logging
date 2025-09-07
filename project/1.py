import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示汉字
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def get_color_features(image_path, n_clusters=8):
    """
    使用 K-means 提取图片的颜色特征（聚类中心）。
    返回一个包含所有聚类中心 RGB 值的列表。
    """
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
        return kmeans.cluster_centers_
    except Exception as e:
        print(f"处理图片 '{image_path}' 时出错: {e}")
        return None


def build_dataset_and_train_model(base_path='dataset/train', n_clusters=8):
    """
    从文件夹构建数据集，并训练 KNN 颜色识别模型。
    """
    X_features = []  # 特征数据 (K-means 聚类中心)
    y_labels = []  # 标签数据 (文件夹名)
    print("--- 开始加载和提取颜色特征 ---")
    for color_name in os.listdir(base_path):
        color_folder_path = os.path.join(base_path, color_name)
        if os.path.isdir(color_folder_path):
            print(f"正在处理 '{color_name}' 文件夹...")
            for image_file in os.listdir(color_folder_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    image_path = os.path.join(color_folder_path, image_file)
                    features = get_color_features(image_path, n_clusters)
                    if features is not None:
                        for feature in features:
                            X_features.append(feature)
                            y_labels.append(color_name)

    print("\n--- 数据加载完成，开始训练模型 ---")
    X = np.array(X_features)
    y = np.array(y_labels)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X, y)
    print("模型训练成功！")
    return knn_model


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


def plot_color_histogram(top_colors):
    """
    使用直方图展示占比最高的前5个颜色。
    """
    if not top_colors:
        return
    percentages = [c['percentage'] for c in top_colors]
    rgb_values = [c['rgb'] for c in top_colors]
    # 将 RGB 值转换为更具可读性的标签
    color_labels = [f"({r}, {g}, {b})" for r, g, b in rgb_values]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(color_labels, percentages, color=[f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb_values])
    plt.xlabel('颜色 (RGB值)')
    plt.ylabel('占比 (%)')
    plt.title('图片颜色占比直方图 (前5大)')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}%', ha='center', va='bottom')
    plt.show()


if __name__ == '__main__':
    model_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/project/model/color_classifier_knn.joblib"
    if os.path.exists(model_path):
        print("--- 正在加载已存在的模型 ---")
        knn_model = joblib.load(model_path)
        print("模型加载成功！")
    else:
        print("--- 模型文件不存在，开始训练新模型 ---")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        knn_model = build_dataset_and_train_model()
        joblib.dump(knn_model, model_path)
        print(f"模型已保存到 '{model_path}'")

    # 预测并显示结果
    if knn_model:
        test_image_path = "C:/Users/28162/Desktop/标注/灰白色中砂岩-20/1217.00-灰白色中砂岩-庆阳仪器-白光-小视野.webp"
        if os.path.exists(test_image_path):
            predicted_color, top_5_colors = predict_image_color(knn_model, test_image_path)
            if top_5_colors:
                print(f"该图片的主要颜色是 '{predicted_color}'")
                print("图片中占比最高的5种颜色及其占比：")
                for i, color_data in enumerate(top_5_colors):
                    print(f"{i + 1}. RGB: {color_data['rgb']}, 占比: {color_data['percentage']:.2f}%")
                plot_color_histogram(top_5_colors)  # 生成并显示直方图
            else:
                print("无法获取颜色数据。")
        else:
            print(f"错误：测试图片 '{test_image_path}' 不存在。")
