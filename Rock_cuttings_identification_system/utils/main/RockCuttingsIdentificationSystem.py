import cv2
import numpy as np
import os
from utils.Clustering.kmeansRGBHSV import clusterKMeans
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
        self.num_classes = 8
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.clusterer = clusterKMeans(self.n_clusters)
        self.detector = MobileNetV3Detect(self.model_path, self.num_classes)
        self.segmenter = ImageSegmentation()
        self.analyzer = RockAnalysis(self.confidence_threshold)

        # 结果保存目录（相对于项目根的 result 文件夹），自动创建
        self.result_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '紫红色泥岩')
        )
        os.makedirs(self.result_dir, exist_ok=True)
    
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
            dpi = 100  # 这里可以换成 150、200 等
            figsize = (width_px / dpi, height_px / dpi)

            for i, seg in enumerate(segmented_images):
                # 控制显示大小：限制最大宽度为 10 英寸、最大高度为 8 英寸，保持原图长宽比，
                # 使用 suptitle 并通过 tight_layout(rect=...) 留出上方空间以确保标题可见
                h_img, w_img = seg.shape[:2]
                max_w, max_h = 10.0, 8.0
                # 计算合适的显示宽高（以英寸为单位，dpi 在函数上方已定义）
                fig_w = min(max_w, max(2.0, w_img / dpi))
                fig_h = min(max_h, max(2.0, (h_img / w_img) * fig_w))
                plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
                plt.imshow(seg.astype("uint8"))
                plt.axis("off")
                plt.suptitle(f"岩性提取结果 {i}", fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()


                # """
                # 保存分割图像到 self.result_dir：
                # - 基于传入的 image_name（或 image_path 的文件名）生成文件名
                # - 处理不同 dtype / 通道数（RGB/ RGBA -> BGR），确保能被保存
                # - 若 cv2.imwrite 失败，使用 cv2.imencode + open(...) 回退保存（支持 Unicode 路径）
                # """
                # base_name = image_name if image_name else os.path.splitext(os.path.basename(image_path))[0]
                # safe_base = "".join(c if c.isalnum() or c in "-_." else "_" for c in base_name)
                # out_name = f"{safe_base}_segmented_{i}.png"
                # out_path = os.path.join(self.result_dir, out_name)

                # try:
                #     seg_uint8 = segmented_images[i]
                #     if seg_uint8.dtype != np.uint8:
                #         seg_uint8 = np.clip(seg_uint8, 0, 255).astype(np.uint8)

                #     # 处理通道：RGB->BGR，RGBA->RGB->BGR，单通道直接写入
                #     if seg_uint8.ndim == 3:
                #         if seg_uint8.shape[2] == 3:
                #             save_img = cv2.cvtColor(seg_uint8, cv2.COLOR_RGB2BGR)
                #         elif seg_uint8.shape[2] == 4:
                #             # RGBA -> RGB -> BGR（丢弃 alpha）
                #             rgb = cv2.cvtColor(seg_uint8, cv2.COLOR_RGBA2RGB)
                #             save_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                #         else:
                #             save_img = seg_uint8
                #     else:
                #         save_img = seg_uint8

                #     # 确保内存连续
                #     save_img = np.ascontiguousarray(save_img)

                #     # 回退：使用 imencode + Python 写入（通常可绕过 OpenCV 的路径/编码问题）
                #     ext = os.path.splitext(out_path)[1] or '.png'
                #     success, buf = cv2.imencode(ext, save_img)
                #     if success:
                #         with open(out_path, 'wb') as f:
                #             f.write(buf.tobytes())
                #         # print(f"已保存分割图像: {out_path}")

                # except Exception as e:
                #     print(f"保存分割图像时出错 ({out_path}): {e}")
                
            # 计算每个聚类的像素占比
            cluster_ratios = self.segmenter.calculate_cluster_ratio(
                cluster_labels, n_clusters=self.n_clusters
            )
            # print("聚类像素占比 ",cluster_ratios)

            # 步骤3: 对三张子图像进行MobileNetV3识别
            # print("步骤3: 进行岩性识别...")
            detection_results = []
            for i, seg_image in enumerate(segmented_images):
                result = self.detector.detect(seg_image)
                detection_results.append(result)
                print(f"  聚类{i+1}: {result['predicted_class']} (置信度: {result['confidence']:.3f}) 占比: {cluster_ratios[i]:.2%}")
            
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

