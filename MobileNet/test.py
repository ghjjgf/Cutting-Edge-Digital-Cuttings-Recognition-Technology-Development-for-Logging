import os
import json
import torch
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans
from model.nets.model_v3 import mobilenet_v3_small

# 图像预处理
def image_preprocessing(image, device):
    """输入路径或PIL对象 -> 标准化Tensor"""
    if isinstance(image, str):  # 路径
        image = Image.open(image).convert("RGB")
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = data_transform(image)
    return torch.unsqueeze(img, dim=0).to(device)

# 模型加载
def load_model_and_classes(mobile_model_path, knn_model_path, class_map_path, device):
    """加载 MobileNet knn 模型"""
    mobile_model = mobilenet_v3_small(num_classes=7).to(device)
    knn_model = joblib.load(knn_model_path)
    if mobile_model and knn_model:
        print("--- 模型加载成功 ---")
    mobile_model.load_state_dict(torch.load(mobile_model_path, map_location=device))
    mobile_model.eval()
    with open(class_map_path, "r") as f:
        class_indict = json.load(f)
    print(class_indict)
    return mobile_model, knn_model, class_indict

# 单张图像预测
def predict_single_image(image, mobile_model, class_indict, device):
    try:
        img_tensor = image_preprocessing(image, device)
        with torch.no_grad():
            output = torch.squeeze(mobile_model(img_tensor)).cpu()
            probs = torch.softmax(output, dim=0)
            idx = torch.argmax(probs).numpy()
        return class_indict[str(idx)], probs[idx].item()
    except Exception as e:
        print(f"[错误] 图像预测失败: {e}")
        return None, None


# 文件夹批量预测
def predict_images_in_folder(folder_path, mobile_model, class_indict, device):
    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".webp")):
                img_path = os.path.join(root, file)
                label, prob = predict_single_image(img_path, mobile_model, class_indict, device)
                if label:
                    results.append([file, label, prob])
    return results

# 视频逐帧预测
def predict_video_frames(video_path, model, class_indict, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[错误] 视频打开失败")
        return []

    results = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label, prob = predict_single_image(img, model, class_indict, device)
        results.append((frame_idx, label, prob))
        print(f"Frame {frame_idx} -> {label} ({prob:.3f})")
        frame_idx += 1
    cap.release()
    return results


# KNN颜色分析
def knn_color_predict(image, knn_model, n_clusters=8):
    try:
        color_percentage = {}

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        img_array = np.array(image).reshape(-1, 3)

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=0,
            n_init="auto", batch_size=4096
        )
        kmeans.fit(img_array)

        # 模拟KNN推理与KMeans聚类中心
        predictions = knn_model.predict(kmeans.cluster_centers_)
        print(f"predictions: {predictions}")
        # 统计每个类别的数量
        unique, counts = np.unique(predictions, return_counts=True)

        # 计算每个类别的占比
        total_count = len(predictions)
        class_ratios = counts / total_count  # 类别占比 = 类别数量 / 总数

        # 获取颜色与类别对应的映射
        class_to_colors = {cls: [] for cls in unique}   
        for i, color in enumerate(kmeans.cluster_centers_):
            class_to_colors[predictions[i]].append(color.astype(int))
        # print(f"class_to_colors init: {class_to_colors}")

        # 统计 top8 主色占比
        counts_pixels = np.bincount(kmeans.labels_)
        top_indices = np.argsort(counts_pixels)[::-1][:8]
        total_pixels = len(img_array)
        
        top_colors = [
            {"rgb": tuple(kmeans.cluster_centers_[i].astype(int)),
             "percentage": (counts_pixels[i] / total_pixels) * 100}
            for i in top_indices
        ]
        # print(f"top_colors: {top_colors}")

        # 遍历 class_to_colors 中的每个类别
        for color_name, color_arrays in class_to_colors.items():
            # 初始化该类别的总占比
            total_percentage = 0
            
            # 遍历该类别中的所有颜色
            for color_array in color_arrays:
                # 转换颜色数组为元组，以便与 top_colors 中的 rgb 进行匹配
                color_tuple = tuple(color_array)
                
                # 查找匹配的颜色并累计 percentage
                for top_color in top_colors:
                    if color_tuple == top_color['rgb']:
                        total_percentage += top_color['percentage']

            # 将每个类别及其总占比添加到 color_percentage 字典中
            color_percentage[color_name] = total_percentage

        # 输出类别与其对应的总占比
        print("类别与其对应的总占比:")
        for color_name, total_percentage in color_percentage.items():
            print(f"类别 {color_name}: 总占比 {total_percentage:.4f}%")

        # print(f"color_percentage: {color_percentage}")

        # 输出占比最高的类别
        most_common = max(color_percentage, key=color_percentage.get)
        most_common_ratio = color_percentage[most_common]

        return most_common, most_common_ratio, color_percentage
    except Exception as e:
        print(f"[错误] KNN颜色分析失败: {e}")
        return None, []


# 过渡层检测
def detect_transition_layer(prev_class, curr_class, curr_colors, prev_color_percentage, color_percentage, cnt, transition_flag):
    # 检测到疑似过渡层的次数 达到5次以上且出现主要岩性变换后，且新主要岩性的百分含量不断上升，则判定为过渡层
    if prev_class is None or curr_class is None:
        return transition_flag, cnt
    elif prev_class == curr_class and color_percentage[curr_colors] < prev_color_percentage[curr_colors]:
        print(f"主要岩性 {curr_class} 百分含量下降")
        return transition_flag, cnt
    elif prev_class == curr_class and color_percentage[curr_colors] > prev_color_percentage[curr_colors]:
        print(f"主要岩性 {curr_class} 百分含量上升")
        return transition_flag, cnt
    elif cnt >= 5 and prev_class != curr_class:
        print(f"[疑似过渡层] 岩性变化 {prev_class} -> {curr_class}")
        transition_flag = True
        return transition_flag, cnt
    elif cnt >= 10 and transition_flag == True:
        print(f"[确认过渡层] 岩性变化为{curr_class}")
        return transition_flag, cnt
    else:
        return transition_flag, cnt

# 更新颜色与岩性映射
def update_curr_colors_and_class(curr_colors, curr_class):
    """
    更新当前的颜色和岩性类别
    
    curr_colors: 当前颜色列表（可能是图像中提取的颜色）
    curr_class: 当前岩性类别列表
    
    cuttings_to_color: 岩性到颜色的映射
    color_to_cuttings: 颜色到岩性的映射
    """
    # 颜色映射 MobileNetV3
    cuttings_to_color = {
        "Dark-gray-mudstone": "灰色",
        "Dark-gray-sandy-mudstone": "灰色",
        "Gray-green-mudstone": "灰绿色",
        "Gray-mudstone-sandstone": "灰色",
        "Light-gray-medium-sandstone": "灰色",
        "Pale-gray-medium-sandstone": "灰白色",
        "Purple-red-mudstone": "紫红色"
    }
    # knn
    color_to_cuttings = {
        "浅灰色": "灰色",
        "灰白色": "灰白色",
        "深灰色": "灰色",
        "深灰色": "灰色",
        "灰绿色": "灰绿色",
        "灰色": "灰色",
        "紫红色": "紫红色"
    }
    # 更新颜色
    if curr_colors in color_to_cuttings:
        updated_colors = color_to_cuttings[curr_colors]
    if curr_class in cuttings_to_color:
        updated_class = cuttings_to_color[curr_class] 

    return updated_colors, updated_class

def update_name_in_dir(frame_save_path, cnt, curr_class):
    # 更新文件夹中最近添加的cnt张图片的岩性名称为curr_class
    if not os.path.exists(frame_save_path):
        print(f"[错误] 文件夹 {frame_save_path} 不存在")
        return
    files = sorted([f for f in os.listdir(frame_save_path) if os.path.isfile(os.path.join(frame_save_path, f))])
    if len(files) < cnt:
        print(f"[警告] 文件夹中图片数量少于 {cnt} 张，无法全部更新")
        cnt = len(files)
    for file in files[-cnt:]:
        old_path = os.path.join(frame_save_path, file)
        new_name = f"{curr_class}_{file.split('_', 1)[-1]}"
        new_path = os.path.join(frame_save_path, new_name)
        os.rename(old_path, new_path)
        print(f"已更新文件名: {old_path} -> {new_path}")

def transition_layers(video_path, knn_model, MobileNetmodel, class_indict, device, frame_save_path="transition_frames"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[错误] 视频打开失败")
        return []
    # 预测结果存储
    results = []
    prev_class, prev_colors = None, None
    frame_idx = 0
    cnt = 0
    prev_color_percentage = None
    transition_flag = False
    update_name = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # MobileNetV3 推理
        curr_class, _ = predict_single_image(img, MobileNetmodel, class_indict, device)
        print(f"mobileinfer {frame_idx} -> {curr_class}")
        # kmeans 聚类推理
        curr_colors, color_ratio, color_percentage = knn_color_predict(img, knn_model, n_clusters=8)
        print(f"\n最常见的类别是: {curr_colors}, 占比: {color_ratio:.4f}")

        # 根据颜色映射 更新curr_class
        updated_colors, updated_classes_color = update_curr_colors_and_class(curr_colors, curr_class)
        # 判断两种模型识别出的颜色是否一致
        if updated_colors == updated_classes_color:
            print(f"MobileNetV3推理的主要岩性: {curr_class}, KNN推理的主要颜色: {curr_colors}, 该岩性主要百分含量: {color_ratio:.4f}\n")
        else:
            color_ratio = color_percentage[updated_classes_color]  # 以MobileNetV3的颜色为准
            print(f"[警告两个模型推理的颜色不一致 其岩性主要百分含量已更新为MobileNet推理的] MobileNetV3推理的主要岩性: {curr_class}, KNN推理的主要颜色: {updated_classes_color}, 该岩性主要百分含量: {color_ratio:.4f}\n")
        # 记录结果
        results.append((frame_idx, curr_class, color_ratio))
        # 检测到疑似过渡层的次数 达到10次以上且出现主要岩性变换后，且新主要岩性的百分含量不断上升，则判定为过渡层
        transition_flag, cnt = detect_transition_layer(prev_class, curr_class, updated_classes_color, prev_color_percentage, color_percentage, cnt, transition_flag)
        if not transition_flag and cnt > 5:
            print(f"主要岩性百分含量下降，疑似过渡层{cnt}次")
            
        elif transition_flag and cnt <= 10:
            print(f"[疑似过渡层] 检测到疑似过渡层{cnt}次 at frame {frame_idx}")
        elif transition_flag and cnt >= 10:
            print(f"[确认过渡层] 检测到过渡层 at frame {frame_idx}")
            update_name_in_dir(frame_save_path, cnt, curr_class)
            transition_flag = False
            cnt = 0

        prev_class = curr_class
        prev_color_percentage = color_percentage
        frame_idx += 1

    cap.release()
    return results


if __name__ == "__main__":
    detect_type = "video"   # 'video' 'image_folder' 'image'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 路径配置
    mobilenet_model_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/model/train/cuttings_MobileNetV3.pth"
    class_map_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/dataset/cuttings_classes.json"
    knn_model_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/project/model/color_classifier_knn.joblib"
    video_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/cuttings_video/cuttings_test.mp4"
    img_folder_path = "path_to_image_folder"
    image_path = "D:/计算机视觉学习/deep-learning-for-image-processing-master/data_set/cuttings/train/Light-gray-medium-sandstone/Light-gray-medium-sandstone_8.webp"
    frame_save_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/frame_save_dir" # 保存过渡层帧的文件夹

    # 加载模型
    MobileNetmodel, knn_model, class_indict = load_model_and_classes(mobilenet_model_path, knn_model_path, class_map_path, device)

    if detect_type == "video":
        transition_layers(video_path, knn_model, MobileNetmodel, class_indict, device)
    elif detect_type == "image_folder":
        results = predict_images_in_folder(img_folder_path, MobileNetmodel, class_indict, device, frame_save_path)
        pd.DataFrame(results, columns=["图像名称", "岩性类别", "概率"]).to_excel("推理结果.xlsx", index=False)
        print("结果已保存到: 推理结果.xlsx")
    elif detect_type == "image":
        label, prob = predict_single_image(image_path, MobileNetmodel, class_indict, device)
        print(f"{os.path.basename(image_path)} -> {label} ({prob:.3f})")

