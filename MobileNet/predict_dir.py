import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd  # 用于生成Excel表格

from model_v3 import mobilenet_v3_small

# 类别名称
'''
    "0": "深灰色泥岩",
    "1": "深灰色粉砂质泥岩",
    "2": "灰绿色泥岩",
    "3": "灰色泥质粉砂岩",
    "4": "浅灰色中砂岩",
    "5": "灰白色中砂岩",
    "6": "紫红色泥岩"
'''

def main():
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


if __name__ == '__main__':
    main()
