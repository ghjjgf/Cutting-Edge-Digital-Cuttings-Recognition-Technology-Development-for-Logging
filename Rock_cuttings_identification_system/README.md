# 岩屑识别系统

基于KMeans聚类和MobileNetV3深度学习的岩屑识别系统，能够自动识别岩屑图像中的岩性类型。

## 系统架构

```
岩屑识别系统/
├── main.py                              # 主程序入口
├── example_usage.py                     # 使用示例
├── requirements.txt                     # 依赖包列表
├── README.md                           # 说明文档
└── utils/                              # 工具模块
    ├── Clustering/
    │   └── kmeans.py                   # KMeans聚类模块
    ├── MobileNet/
    │   └── MobileNetV3_detect.py       # MobileNetV3识别模块
    ├── ImageSegmentation/
    │   └── image_segmentation.py       # 图像分割模块
    └── RockAnalysis/
        └── rock_analysis.py            # 岩性分析模块
```

## 功能特性

### 1. 图像聚类
- 使用KMeans算法对输入图像进行聚类
- 自动识别三个聚簇中心
- 支持彩色图像处理

### 2. 图像分割
- 根据聚类结果将原图分割为三张子图像
- 每张子图像只保留对应聚簇的像素
- 非目标聚簇像素置为黑色

### 3. 岩性识别
- 使用MobileNetV3深度学习模型
- 支持10种常见岩性类型识别
- 输出置信度分数

### 4. 智能分析
- 置信度过滤（默认20%阈值）
- 有效聚簇中心判定
- 岩性占比计算
- 复合定名支持（预留接口）

## 岩性类别

系统支持以下岩性类型识别：
- 泥岩
- 砂岩
- 粉砂岩
- 砾岩
- 石灰岩
- 白云岩
- 页岩
- 片岩
- 花岗岩
- 玄武岩

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 单张图像识别

```python
from main import RockCuttingsIdentificationSystem

# 创建系统实例
system = RockCuttingsIdentificationSystem()

# 识别单张图像
result = system.identify_rock_cuttings("path/to/your/image.jpg")

# 查看结果
print(f"主要岩性: {result['primary_rock']}")
print(f"占比: {result['ratio']:.2%}")
print(f"置信度: {result['confidence']:.3f}")
```

### 2. 批量处理

```python
# 批量处理文件夹中的所有图像
results = system.batch_identify("path/to/image/folder", "results.json")
```

### 3. 自定义置信度阈值

```python
# 设置置信度阈值为30%
system = RockCuttingsIdentificationSystem(confidence_threshold=0.3)
```

## 输出结果

系统返回的结果包含以下信息：

```python
{
    'status': 'single_rock',           # 识别状态
    'message': '该图像为砂岩',          # 结果描述
    'primary_rock': '砂岩',            # 主要岩性
    'ratio': 0.65,                     # 占比
    'confidence': 0.89,                # 置信度
    'composition': [...],              # 详细组成
    'image_name': 'image.jpg',         # 图像名称
    'cluster_ratios': [0.65, 0.25, 0.10]  # 各聚类占比
}
```

### 状态说明

- `unrecognizable`: 无法识别（所有聚类置信度都低于阈值）
- `single_rock`: 单一岩性（只有一个有效聚类）
- `dominant_rock`: 主导岩性（有占比>50%的岩性）
- `complex_naming`: 需要复合定名（多个岩性且都<50%占比）

## 处理流程

1. **图像聚类**: 对输入图像进行KMeans聚类，得到三个聚簇中心
2. **图像分割**: 根据聚类标签将原图分割为三张子图像
3. **岩性识别**: 将三张子图像分别输入MobileNetV3进行识别
4. **置信度过滤**: 过滤掉置信度<20%的无效聚类
5. **岩性判定**: 根据有效聚类结果进行最终岩性判定

## 注意事项

1. **模型训练**: 当前使用的是预训练模型，建议根据实际岩屑数据重新训练
2. **类别调整**: 岩性类别可根据实际需求在`MobileNetV3_detect.py`中修改
3. **复合定名**: 当前版本暂不支持复合定名法，需要进一步开发
4. **图像格式**: 支持常见图像格式（JPG、PNG、BMP、TIFF等）

## 运行示例

```bash
# 运行主程序
python main.py

# 运行使用示例
python example_usage.py
```

## 技术栈

- **深度学习**: PyTorch + MobileNetV3
- **图像处理**: OpenCV
- **聚类算法**: scikit-learn KMeans
- **数值计算**: NumPy

## 开发说明

系统采用模块化设计，各功能模块独立，便于维护和扩展：

- `clusterKMeans`: 负责图像聚类
- `MobileNetV3Detect`: 负责岩性识别
- `ImageSegmentation`: 负责图像分割
- `RockAnalysis`: 负责结果分析和判定

## 许可证

本项目仅供学习和研究使用。
