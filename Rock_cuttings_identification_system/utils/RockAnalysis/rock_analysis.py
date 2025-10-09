import numpy as np
from typing import List, Dict, Tuple


class RockAnalysis():
    def __init__(self, confidence_threshold=0.2):
        """
        初始化岩性分析器
        Args:
            confidence_threshold: 置信度阈值，默认20%
        """
        self.confidence_threshold = confidence_threshold
    
    def filter_valid_clusters(self, detection_results: List[Dict], cluster_ratios: List[float]) -> List[List]:
        """
        过滤有效聚簇中心，保留置信度大于阈值的岩性
        Args:
            detection_results: MobileNetV3检测结果列表
            cluster_ratios: 每个聚类的像素占比列表
        Returns:
            cuttings: 有效聚簇中心信息，格式为[原图名称, 类别, 占比, 置信度]
        """
        cuttings = []
        
        for i, (result, ratio) in enumerate(zip(detection_results, cluster_ratios)):
            # 检查是否有岩性置信度大于阈值
            max_confidence = result['confidence']
            
            if max_confidence >= self.confidence_threshold:
                # 有效聚簇中心
                predicted_class = result['predicted_class']
                cuttings.append([f"cluster_{i}", predicted_class, ratio, max_confidence])
        
        return cuttings
    
    def analyze_rock_composition(self, cuttings: List[List]) -> Dict:
        """
        分析岩性组成，进行最终判定
        Args:
            cuttings: 有效聚簇中心信息列表
        Returns:
            analysis_result: 分析结果字典
        """
        if not cuttings:
            # 二维列表为空，无法识别
            return {
                'status': 'unrecognizable',
                'message': '该图像无法采取该方法进行岩性识别',
                'primary_rock': None,
                'ratio': None,
                'confidence': None,
                'composition': []
            }
        
        elif len(cuttings) == 1:
            # 只有一个有效聚簇中心
            rock_name, ratio, confidence = cuttings[0][1], cuttings[0][2], cuttings[0][3]
            return {
                'status': 'single_rock',
                'message': f'该图像为{rock_name}',
                'primary_rock': rock_name,
                'ratio': ratio,
                'confidence': confidence,
                'composition': cuttings
            }
        
        else:
            # 多个有效聚簇中心，需要进一步判断
            return self._analyze_multiple_rocks(cuttings)
    
    def _analyze_multiple_rocks(self, cuttings: List[List]) -> Dict:
        """
        分析多个岩性的情况
        Args:
            cuttings: 有效聚簇中心信息列表
        Returns:
            analysis_result: 分析结果字典
        """
        # 按占比排序
        sorted_cuttings = sorted(cuttings, key=lambda x: x[2], reverse=True)
        
        # 检查是否有占比大于50%的岩性
        max_ratio = sorted_cuttings[0][2]
        if max_ratio > 0.5:
            # 直接判定为主要岩性
            primary_rock = sorted_cuttings[0][1]
            confidence = sorted_cuttings[0][3]
            
            return {
                'status': 'dominant_rock',
                'message': f'该图像主要为{primary_rock}（占比{max_ratio:.2%}）',
                'primary_rock': primary_rock,
                'ratio': max_ratio,
                'confidence': confidence,
                'composition': sorted_cuttings
            }
        else:
            # 需要复合定名法（暂不实现）
            return {
                'status': 'complex_naming',
                'message': '需要采用复合定名法，但当前版本暂不支持',
                'primary_rock': None,
                'ratio': None,
                'confidence': None,
                'composition': sorted_cuttings,
                'note': '如存在一维列表占比皆小于50%，则需采用复合定名法，但MobileNetV3中分类出的岩屑部分为复合命名的岩性，难以再次进行复合定名'
            }
    
    def get_confidence_statistics(self, detection_results: List[Dict]) -> Dict:
        """
        获取置信度统计信息
        Args:
            detection_results: 检测结果列表
        Returns:
            stats: 置信度统计信息
        """
        if not detection_results:
            return {'max_confidence': 0, 'min_confidence': 0, 'avg_confidence': 0}
        
        confidences = [result['confidence'] for result in detection_results]
        
        return {
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'avg_confidence': np.mean(confidences),
            'valid_clusters': sum(1 for c in confidences if c >= self.confidence_threshold)
        }
