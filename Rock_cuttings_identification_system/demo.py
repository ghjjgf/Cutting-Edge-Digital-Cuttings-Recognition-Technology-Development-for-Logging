#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岩屑识别系统
"""
from ast import arg
from utils.main.RockCuttingsIdentificationSystem import RockCuttingsIdentificationSystem
import os


def single_image_example(image_path, model_path, confidence_threshold):
    """单张图像识别示例"""
    print("=== 单张图像识别示例 ===")
    
    # 创建岩屑识别系统
    system = RockCuttingsIdentificationSystem(model_path, confidence_threshold)
    
    if os.path.exists(image_path):
        result = system.identify_rock_cuttings(image_path)
        print(f"\n图像: {result.get('image_name', 'Unknown')}")
        print(f"状态: {result.get('status', 'Unknown')}")
        print(f"结果: {result.get('message', 'No message')}")
        
        if result.get('primary_rock'):
            print(f"主要岩性: {result['primary_rock']}")
            print(f"占比: {result['ratio']:.2%}")
            print(f"置信度: {result['confidence']:.3f}")
        
        if result.get('composition'):
            print("\n详细组成:")
            for i, comp in enumerate(result['composition']):
                print(f"  聚类{comp[0]}: {comp[1]} (占比: {comp[2]:.2%}, 置信度: {comp[3]:.3f})")
    else:
        print(f"图像文件不存在: {image_path}")
        print("请将image_path修改为实际的图像文件路径")

def batch_processing_example(confidence_threshold, image_folder, output_file, model_path):
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 创建岩屑识别系统
    system = RockCuttingsIdentificationSystem(model_path, confidence_threshold)
    
    if os.path.exists(image_folder):
        results = system.batch_identify(image_folder, output_file)
        
        print(f"\n处理完成，共处理 {len(results)} 张图像")
        print(f"结果已保存到: {output_file}")
        
        # 统计结果
        status_counts = {}
        for result in results:
            if 'error' not in result:
                status = result.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\n结果统计:")
        for status, count in status_counts.items():
            print(f"  {status}: {count} 张")
    else:
        print(f"图像文件夹不存在: {image_folder}")
        print("请将image_folder修改为实际的图像文件夹路径")

# def custom_confidence_threshold_example():
#     """自定义置信度阈值示例"""
#     print("\n=== 自定义置信度阈值示例 ===")
    
#     # 使用不同的置信度阈值
#     thresholds = [0.1, 0.2, 0.3, 0.5]
    
#     for threshold in thresholds:
#         print(f"\n置信度阈值: {threshold*100}%")
#         system = RockCuttingsIdentificationSystem(confidence_threshold=threshold)
        
#         # 这里可以添加实际的图像处理代码
#         print(f"系统已配置置信度阈值为 {threshold*100}%")

if __name__ == "__main__":
    print("启动岩屑识别系统")
    print("=" * 50)
    
    # 运行示例
    args = {
        "confidence_threshold": 0.2,
        #"image_path": "C:/Users/28162/Desktop/Rock_cuttings_identification_system/val/Purplish-red mudstone.webp",
        #"image_path": "H:/dataset/深灰色粉砂质泥岩/A/1352.00-深灰色粉砂质泥岩-悦84井-庆阳仪器-白光-小视野.webp",
        "image_path": "C:/Users/28162/Desktop/Rock_cuttings_identification_system/resized_image_0.5.jpg",
        "image_folder": "path/to/your/image/folder",
        "model_path": "C:/Users/28162/Desktop/Rock_cuttings_identification_system/utils/MobileNet/model/train/cuttings_MobileNetV3.pth",
    }

    single_image_example(args["image_path"], args["model_path"], args["confidence_threshold"])
    # batch_processing_example(args["confidence_threshold"], args["image_folder"], args["output_file"], args["model_path"])
    # custom_confidence_threshold_example(confidence_threshold=0.2)