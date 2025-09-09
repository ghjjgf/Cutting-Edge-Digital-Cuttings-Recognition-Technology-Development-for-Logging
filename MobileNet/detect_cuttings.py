import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd  # 用于生成Excel表格
from MobileNet.model.nets.model_v3 import mobilenet_v3_small
import cv2
from sklearn.cluster import MiniBatchKMeans
import joblib
import numpy as np
'''

# 图像
def MobileNet_detect_cuttings_images():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "D:/计算机视觉学习/deep-learning-for-image-processing-master/data_set/cuttings/val/Dark-gray-sandy-mudstone/Dark-gray-sandy-mudstone_1.webp"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/cuttings_classes.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = mobilenet_v3_small(num_classes=7).to(device)
    # load model weights
    model_weight_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/model/train/cuttings_MobileNetV3.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

# 文件夹
def MobileNet_detect_cuttings_dir():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图像预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 输入图像文件夹路径
    img_folder_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/MCcode/MC_crop"
    assert os.path.exists(img_folder_path), f"文件夹: '{img_folder_path}' 不存在."

    # 读取类的名称映射
    json_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/cuttings_classes.json"
    assert os.path.exists(json_path), f"文件: '{json_path}' 不存在."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 加载模型
    model = mobilenet_v3_small(num_classes=7).to(device)
    model_weight_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/model/train/cuttings_MobileNetV3.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 准备存储推理结果
    results = []

    # 遍历图像文件夹
    for root, dirs, files in os.walk(img_folder_path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png", ".webp")):  # 仅处理图片格式
                img_path = os.path.join(root, file)
                img = Image.open(img_path)

                # 进行预处理
                img_tensor = data_transform(img)
                img_tensor = torch.unsqueeze(img_tensor, dim=0)

                # 推理
                with torch.no_grad():
                    output = torch.squeeze(model(img_tensor.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()

                # 获取推理结果
                predict_label = class_indict[str(predict_cla)]
                results.append([file, predict_label])  # 记录文件名和推理结果

    # 保存结果到Excel文件
    df = pd.DataFrame(results, columns=["图像名称", "推理结果"])
    excel_path = "推理结果.xlsx"
    df.to_excel(excel_path, index=False)

    print(f"推理结果已保存到: {excel_path}")
'''

# 视频
def MobileNet_detect_cuttings_video(video_path):
    # 设置设备（GPU或CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图像预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载类名映射
    json_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/cuttings_classes.json"
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # 创建MobileNetV3模型并加载权重
    model = mobilenet_v3_small(num_classes=7).to(device)
    model_weight_path = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/MobileNet/model/train/cuttings_MobileNetV3.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 视频播放完毕

        # 将图像从BGR转为RGB并进行预处理
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0).to(device)

        # 推理
        with torch.no_grad():
            output = torch.squeeze(model(img)).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # 显示推理结果
        print_res = f"Frame {frame_idx} - class: {class_indict[str(predict_cla)]}   prob: {predict[predict_cla].numpy():.3f}"
        print(print_res)

        # 在图像上显示推理结果
        plt.figure()
        plt.imshow(frame)
        plt.title(print_res)
        plt.show()

        # 增加帧索引
        frame_idx += 1

    cap.release()  # 释放视频文件资源

# KNN颜色聚类+预测
def Knn_detect_cuttings(image_path, model_path, n_clusters=8):
    """
    使用KNN模型预测图像颜色，并返回岩性分类及前五大颜色的占比
    """
    try:
        # 加载 KNN 模型
        if os.path.exists(model_path):
            print("--- 正在加载已存在的模型 ---")
            knn_model = joblib.load(model_path)
            print("模型加载成功！")
        else:
            print(f"错误：模型文件 '{model_path}' 不存在。请先训练模型。")
            return None, None
        
        print(f"\n--- 正在预测图片: {os.path.basename(image_path)} ---")
        # 读取图片
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        reshaped_img = img_array.reshape(-1, 3)

        # 使用MiniBatchKMeans进行颜色聚类
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init='auto',
            batch_size=4096
        )
        kmeans.fit(reshaped_img)
        cluster_centers = kmeans.cluster_centers_

        # 预测颜色类别
        predictions = knn_model.predict(cluster_centers)
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        most_common_prediction = unique_predictions[np.argmax(counts)]

        # 获取最常见的颜色和每个颜色的像素占比
        labels = kmeans.labels_
        counts_pixels = np.bincount(labels)
        top_5_indices = np.argsort(counts_pixels)[::-1][:5]

        top_5_colors = []
        total_pixels = len(reshaped_img)
        for index in top_5_indices:
            color_rgb = tuple(cluster_centers[index].astype(int))
            percentage = (counts_pixels[index] / total_pixels) * 100
            top_5_colors.append({'rgb': color_rgb, 'percentage': percentage})

        # 输出颜色预测结果
        print(f"该图片的主要颜色是 '{most_common_prediction}'")
        print("图片中占比最高的5种颜色及其占比：")
        for i, color_data in enumerate(top_5_colors):
            print(f"{i + 1}. RGB: {color_data['rgb']}, 占比: {color_data['percentage']:.2f}%")
        
        return most_common_prediction, top_5_colors

    except Exception as e:
        print(f"处理图片 '{image_path}' 时出错: {e}")
        return None, None

def map_class_indices():
    rock_color_mapping = {
        'light_gray_sandstone': ['gray', 'light_gray', 'dark_gray'],
        'purple_red_mudstone': ['purple_red']
    }
    return rock_color_mapping

def transition_layers():
    # 参数加载 --计算资源、类别映射颜色map、模型路径、视频路径等

    # MobileNet岩性识别 --return cuttings_name

    # KNN颜色预测 --return most_common_prediction, top_5_colors

    # 保存每一帧预测结果于新文件夹中 并对比两种模型预测出的颜色是否一致 --一致：输出岩性+颜色占比；不一致：输出岩性+颜色占比

    # 过渡层判别：根据颜色占比来判断是否为过渡层 KNN预测的颜色占比中，如存在某占比并非最大的颜色的占比一直上升，而原先某种占比最大的颜色的占比一直下降
    # 则判断为过渡层，更新新文件夹种所有的图片的岩性为颜色占比一直上升的岩性名字

    # 结果展示 --可视化视频+预测结果


    pass

if __name__ == '__main__':
    transition_layers()