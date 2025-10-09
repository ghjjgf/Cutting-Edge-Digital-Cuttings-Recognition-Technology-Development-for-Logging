import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib
import os
import cv2
from tqdm import tqdm  # 添加进度条库

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示汉字
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def get_image_avg_features(image_path):
    """
    计算图片的平均 RGB 和平均 H 值作为特征向量。
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array_rgb = np.array(img)
        img_array_hsv = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2HSV)

        avg_r, avg_g, avg_b = np.mean(img_array_rgb, axis=(0, 1))
        avg_h = np.mean(img_array_hsv[:, :, 0])

        return np.array([avg_r, avg_g, avg_b, avg_h])

    except Exception as e:
        print(f"处理图片 '{image_path}' 时出错: {e}")
        return None


def build_dataset_and_train_model(base_path='dataset/train'):
    """
    从文件夹构建数据集，并训练 KNN 颜色识别模型。
    """
    X_features = []
    y_labels = []

    print("--- 开始加载和提取颜色特征 ---")

    for color_name in os.listdir(base_path):
        color_folder_path = os.path.join(base_path, color_name)

        if os.path.isdir(color_folder_path):
            print(f"正在处理 '{color_name}' 文件夹...")

            for image_file in os.listdir(color_folder_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')): 
                    image_path = os.path.join(color_folder_path, image_file)
                    features = get_image_avg_features(image_path)

                    if features is not None:
                        X_features.append(features)
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
        # 第一步：使用均值模型进行主色预测
        avg_features = get_image_avg_features(image_path)
        if avg_features is None:
            return "无法识别", None
        predicted_main_color = model.predict(avg_features.reshape(1, -1))[0]

        # 第二步：使用高维特征（RGB+H）进行K-means聚类，用于直方图
        img = Image.open(image_path).convert('RGB')
        img_array_rgb = np.array(img)
        img_array_hsv = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2HSV)

        # 组合RGB和H通道
        reshaped_img = np.concatenate(
            (img_array_rgb.reshape(-1, 3), img_array_hsv[:, :, 0].reshape(-1, 1)),
            axis=1
        )

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init='auto',
            batch_size=4096
        )
        kmeans.fit(reshaped_img)

        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts_pixels = np.bincount(labels)
        top_5_indices = np.argsort(counts_pixels)[::-1][:5]

        top_5_colors = []
        total_pixels = len(reshaped_img)
        for index in top_5_indices:
            # 提取RGB和H值
            rgb_center = cluster_centers[index, :3].astype(int)
            h_center = cluster_centers[index, 3].astype(int)
            percentage = (counts_pixels[index] / total_pixels) * 100

            top_5_colors.append({
                'rgb': tuple(rgb_center),
                'h': h_center,
                'percentage': percentage
            })

        # 获取前3个颜色的均值
        avg_rgb = np.mean([color['rgb'] for color in top_5_colors[:3]], axis=0)
        avg_h = np.mean([color['h'] for color in top_5_colors[:3]])
        print(f"图片的平均 RGB: {avg_rgb}, 平均 H: {avg_h}")

        return predicted_main_color, top_5_colors, avg_rgb, avg_h

    except Exception as e:
        print(f"处理图片 '{image_path}' 时出错: {e}")
        return "无法识别", None, None, None


def process_image_and_draw(image_path, avg_rgb, avg_h):
    """
    遍历图像中的每个像素，判断其颜色是否与平均颜色接近，若接近则绘制为红色，否则为蓝色。
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    img_array_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # 设置阈值
    threshold_rgb = 20
    threshold_h = 10

    # 遍历所有像素
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            r, g, b = img_array[y, x]
            h = img_array_hsv[y, x, 0]

            # 判断RGB和H值是否接近
            if (abs(r - avg_rgb[0]) <= threshold_rgb and
                abs(g - avg_rgb[1]) <= threshold_rgb and
                abs(b - avg_rgb[2]) <= threshold_rgb and
                abs(h - avg_h) <= threshold_h):
                img_array[y, x] = [255, 0, 0]  # 红色
            else:
                img_array[y, x] = [0, 0, 255]  # 蓝色

    # 保存并展示结果
    result_img = Image.fromarray(img_array)
    result_img.show()
    result_img.save("gray_green_cuttings.png")


if __name__ == '__main__':
    model_path = 'kmeans/model/avg_color_classifier_knn.joblib'

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

    if knn_model:
        test_image_path = "H:/dataset/灰绿色泥岩/A/2325.00-灰绿色泥岩-孟76井-庆阳仪器-白光-小视野.webp"

        if os.path.exists(test_image_path):
            # KNN输出主要颜色，K-means输出前5大颜色及其占比
            predicted_color, top_5_colors, avg_rgb, avg_h = predict_image_color(knn_model, test_image_path, n_clusters=8)

            if top_5_colors:
                print(f"\n预测结果：该图片的主要颜色是 '{predicted_color}'")
                print("图片中占比最高的5种颜色及其占比：")
                for i, color_data in enumerate(top_5_colors):
                    print(f"{i + 1}. RGB: {color_data['rgb']}, H: {color_data['h']}, 占比: {color_data['percentage']:.2f}%")

                # 处理图像并绘制
                process_image_and_draw(test_image_path, avg_rgb, avg_h)
            else:
                print("无法获取颜色数据。")
        else:
            print(f"错误：测试图片 '{test_image_path}' 不存在。")
