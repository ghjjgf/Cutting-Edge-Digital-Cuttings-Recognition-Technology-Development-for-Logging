import cv2
import numpy as np
import os
from utils.Clustering.kmeans import clusterKMeans
from utils.MobileNet.MobileNetV3_detect import MobileNetV3Detect
from utils.ImageSegmentation.image_segmentation import ImageSegmentation
from utils.RockAnalysis.rock_analysis import RockAnalysis
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示汉字
class RockCuttingsIdentificationSystem():
    def __init__(self, model_path=None, confidence_threshold=None):
        """
        初始化岩屑识别系统
        Args:
            model_path: MobileNetV3模型路径
            confidence_threshold: 置信度阈值，默认None
        """
        # 初始化各个模块
        self.n_clusters = 3
        self.num_classes = 7
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.clusterer = clusterKMeans(self.n_clusters)
        self.detector = MobileNetV3Detect(self.model_path, self.num_classes)
        self.segmenter = ImageSegmentation()
        self.analyzer = RockAnalysis(self.confidence_threshold)
    
    def identify_rock_cuttings(self, image_path, image_name=None):
        """
        识别岩屑图像
        Args:
            image_path: 图像文件路径
            image_name: 图像名称（可选）
        Returns:
            result: 识别结果字典
        """

        try:
            # 步骤1: 对原图进行聚类操作
            print("步骤1: 进行KMeans聚类...")
            cluster_centers, cluster_labels = self.clusterer.fit(image_path)
            print("聚类中心 ",cluster_centers)
            print("聚类标签 ",cluster_labels)

            # 步骤2: 根据聚类结果分割图像
            print("步骤2: 分割图像...")
            segmented_images = self.segmenter.segment_by_clusters(
                image_path, cluster_labels, n_clusters=self.n_clusters
            )
            print(f"分割得到 {len(segmented_images)} 张子图像")

            # 目标分辨率
            width_px, height_px = 5496, 3672
            dpi = 150  # 这里可以换成 150、200 等
            figsize = (width_px / dpi, height_px / dpi)

            for i, seg in enumerate(segmented_images):
                plt.figure(figsize=figsize, dpi=dpi)  # 每次单独建一张画布
                plt.imshow(seg.astype("uint8"))
                plt.title(f"岩性提取结果 {i}")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

            # 计算每个聚类的像素占比
            cluster_ratios = self.segmenter.calculate_cluster_ratio(
                cluster_labels, n_clusters=self.n_clusters
            )
            print("聚类像素占比 ",cluster_ratios)

            # 步骤3: 对三张子图像进行MobileNetV3识别
            print("步骤3: 进行岩性识别...")
            detection_results = []
            for i, seg_image in enumerate(segmented_images):
                result = self.detector.detect(seg_image)
                detection_results.append(result)
                print(f"  聚类{i+1}: {result['predicted_class']} (置信度: {result['confidence']:.3f})")
            
            # 步骤4: 过滤有效聚簇中心
            print("步骤4: 过滤有效聚簇中心...")
            cuttings = self.analyzer.filter_valid_clusters(detection_results, cluster_ratios)
            print(f"有效聚簇中心数量: {len(cuttings)}")
            
            # 步骤5: 进行最终岩性判定
            print("步骤5: 进行最终岩性判定...")
            analysis_result = self.analyzer.analyze_rock_composition(cuttings)
            
            # 添加图像信息到结果中
            analysis_result['image_name'] = image_name
            analysis_result['image_path'] = image_path
            analysis_result['cluster_centers'] = cluster_centers.tolist()
            analysis_result['cluster_ratios'] = cluster_ratios
            
            return analysis_result
            
        except Exception as e:
            return {'error': f'处理过程中发生错误: {str(e)}'}
    
    def batch_identify(self, image_folder, output_file=None):
        """
        批量识别文件夹中的图像
        Args:
            image_folder: 图像文件夹路径
            output_file: 输出结果文件路径（可选）
        Returns:
            results: 批量识别结果列表
        """
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        results = []
        
        for filename in os.listdir(image_folder):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                image_path = os.path.join(image_folder, filename)
                print(f"\n处理图像: {filename}")
                
                result = self.identify_rock_cuttings(image_path, filename)
                results.append(result)
                
                # 打印结果摘要
                if 'error' not in result:
                    print(f"识别结果: {result['message']}")
                else:
                    print(f"错误: {result['error']}")
        
        # 保存结果到文件
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    def _save_results(self, results, output_file):
        """保存结果到文件"""
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")

def load_model():
    """加载模型（保持向后兼容）"""
    return MobileNetV3Detect()

