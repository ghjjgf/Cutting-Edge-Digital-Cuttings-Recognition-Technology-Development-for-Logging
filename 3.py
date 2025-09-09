import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import joblib
import os
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示汉字
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def get_color_features(img_source, n_clusters=8):
    """
    使用 K-means 提取图片的颜色特征（聚类中心）。
    输入可以是图片路径或一个RGB格式的NumPy数组。
    返回一个包含所有聚类中心 RGB 值的列表。
    """
    try:
        # 如果输入是图片路径，则打开图片
        if isinstance(img_source, str):
            img = Image.open(img_source).convert('RGB')
            img_array = np.array(img)
        # 如果输入是NumPy数组，则直接使用
        elif isinstance(img_source, np.ndarray):
            img_array = img_source
        else:
            raise TypeError("输入必须是图片路径（字符串）或NumPy数组。")

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
        print(f"处理颜色特征时出错: {e}")
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


def predict_image_color(model, img_source, n_clusters=8):
    """
    使用训练好的模型预测图片或视频帧的颜色，并返回前5大颜色及其占比。
    输入可以是图片路径或一个RGB格式的NumPy数组。
    """
    try:
        if isinstance(img_source, str):
            img = Image.open(img_source).convert('RGB')
            img_array = np.array(img)
            source_name = os.path.basename(img_source)
        elif isinstance(img_source, np.ndarray):
            img_array = img_source
            source_name = "视频帧"
        else:
            raise TypeError("输入必须是图片路径（字符串）或NumPy数组。")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init='auto',
            batch_size=4096
        )
        kmeans.fit(img_array.reshape(-1, 3))
        cluster_centers = kmeans.cluster_centers_
        predictions = model.predict(cluster_centers)
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        most_common_prediction = unique_predictions[np.argmax(counts)]
        labels = kmeans.labels_
        counts_pixels = np.bincount(labels)
        top_5_indices = np.argsort(counts_pixels)[::-1][:5]
        top_5_colors = []
        total_pixels = len(img_array.reshape(-1, 3))
        for index in top_5_indices:
            color_rgb = tuple(cluster_centers[index].astype(int))
            percentage = (counts_pixels[index] / total_pixels) * 100
            top_5_colors.append({'rgb': color_rgb, 'percentage': percentage})
        return most_common_prediction, top_5_colors
    except Exception as e:
        print(f"处理 '{source_name}' 时出错: {e}")
        return "无法识别", None


def process_video(video_path, model, n_clusters=8, interval_sec=0.5):
    """
    每隔指定时间间隔检测一次视频帧的颜色。
    """
    if not os.path.exists(video_path):
        print(f"错误：视频文件 '{video_path}' 不存在。")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{video_path}'。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)
    frame_count = 0

    print(f"--- 正在检测视频: {os.path.basename(video_path)} ---")
    print(f"视频帧率: {fps} FPS, 将每隔 {interval_sec} 秒检测一次。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 检测间隔
        if frame_count % frame_interval == 0:
            # OpenCV 默认读取的是 BGR 格式，需要转换为 RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            most_common, top_colors = predict_image_color(model, rgb_frame, n_clusters)

            if top_colors:
                print(f"第 {frame_count} 帧 ({frame_count / fps:.2f} 秒):")
                print(f"  预测主色为: '{most_common}'")
                print(f"  占比最高的颜色: RGB {top_colors[0]['rgb']}, 占比 {top_colors[0]['percentage']:.2f}%")

        frame_count += 1

    cap.release()
    print("\n视频检测完成。")


# --- 主程序逻辑 ---

if __name__ == '__main__':
    model_path = 'model/color_classifier_knn.joblib'
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
        # 替换为你要检测的视频文件路径
        video_file_path = '10_second_video.mp4'

        # 调用视频处理函数
        process_video(video_file_path, knn_model, n_clusters=8, interval_sec=0.5)
