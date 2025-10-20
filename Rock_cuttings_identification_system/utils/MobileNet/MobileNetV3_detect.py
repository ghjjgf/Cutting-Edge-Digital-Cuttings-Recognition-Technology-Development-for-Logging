import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from utils.MobileNet.model.net.MobileNetV3 import mobilenet_v3_small
from PIL import Image


class MobileNetV3Detect():
    def __init__(self, model_path, num_classes):
        """
        初始化MobileNetV3检测器
        Args:
            model_path: 预训练模型路径
            num_classes: 岩性类别数量
        """
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # 岩性类别名称（示例，实际使用时需要根据训练数据调整）
        self.class_names = [
            "灰白色中砂岩",
            "灰绿色泥岩",
            "灰色泥质粉砂岩",
            "其他",
            "浅灰色中砂岩",
            "深灰色粉砂质泥岩",
            "深灰色泥岩",
            "紫红色泥岩"
        ]
        # 加载模型
        self.model = self._load_model()

    def preprocessing(self, image, device):
        """
        通用预处理函数
        输入: 路径 / PIL.Image / numpy.ndarray (BGR 或 RGB)
        输出: 归一化 tensor (1, C, H, W) 放到指定 device
        """
        # 如果是文件路径
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # 如果是 numpy.ndarray
        elif isinstance(image, np.ndarray):
            # BGR -> RGB
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 转 PIL.Image
            image = Image.fromarray(image)

        # 如果是 PIL.Image，直接使用
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        # 数据变换
        data_transform = transforms.Compose([
            transforms.Resize(640),
            transforms.CenterCrop(640),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])
        img = data_transform(image)
        
        # 增加 batch 维度并放到 device
        return img.unsqueeze(0).to(device)

        
    def _load_model(self):
        """加载MobileNetV3模型"""
        # 使用预训练的MobileNetV3
        model = mobilenet_v3_small(self.num_classes).to(self.device)
        if model:
            print("模型加载成功")
        # 修改最后一层以适应岩性分类
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, self.num_classes)
        
        if self.model_path:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def detect(self, image):
        """
        对输入图像进行岩性识别
        Args:
            image: 输入图像 (numpy array, BGR格式 或 PIL.Image)
        Returns:
            result: 包含岩性名称、置信度、所有类别置信度的字典
        """


        # 如果是 PIL.Image -> 转为 RGB numpy
        if isinstance(image, Image.Image):
            image_rgb = np.array(image.convert('RGB'))
        # 如果是 numpy.ndarray
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 假设输入是 BGR -> 转 RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        # 预处理图像 (self.preprocessing 应返回 torch.Tensor)
        input_tensor = self.preprocessing(image_rgb, self.device)

        # 确保带 batch 维度 (1, C, H, W)
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_scores = probabilities.cpu().numpy()[0]

        # 获取最高置信度类别
        predicted_class_idx = np.argmax(confidence_scores)
        predicted_class = self.class_names[predicted_class_idx]
        max_confidence = confidence_scores[predicted_class_idx]

        # 构建结果字典
        result = {
            'predicted_class': predicted_class,
            'confidence': float(max_confidence),
            'all_confidences': {
                self.class_names[i]: float(confidence_scores[i]) 
                for i in range(len(self.class_names))
            }
        }

        return result

    
    def get_class_names(self):
        """获取所有岩性类别名称"""
        return self.class_names