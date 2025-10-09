import cv2

def resize_image(image, target_size):
    """
    调整图像大小
    Args:
        image: 输入图像
        target_size: 目标大小 (width, height)
    Returns:
        resized_image: 调整大小后的图像
    """
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    #cv2.imshow("Resized Image", resized_image)
    #cv2.waitKey(0)
    cv2.imwrite("resized_image_0.1.jpg", resized_image)
    return resized_image

if __name__ == "__main__":
    # 读取图像
    image_path = "C:\\Users\\28162\\Desktop\\Rock_cuttings_identification_system\\val\\Purplish-red mudstone.webp"  # 替换为实际图像路径
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"无法读取图像: {image_path}")
    else:
        print(f"原始图像大小: {image.shape[1]}x{image.shape[0]}")
        
        # 等比缩小 以防图像失真
        scale_factor = 0.1  # 缩小比例
        target_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        
        # 调整图像大小
        resized_image = resize_image(image, target_size)
        print(f"调整后图像大小: {resized_image.shape[1]}x{resized_image.shape[0]}")
        
        cv2.destroyAllWindows()
