import cv2
import numpy as np
from PIL import Image


# 获取图像所有像素的坐标和RGB值
def get_pixel_data(image_path):
    try:
        img = Image.open(image_path).convert("RGB")  # 保证是RGB
        width, height = img.size
        pixels = img.load()
        pixel_data = []
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                pixel_data.append(f"{x} {y} {r} {g} {b}")
        return pixel_data
    except Exception as e:
        print(f"发生错误：{e}")
        return None


# 将图像像素数据保存到TXT文件
def save_to_txt(image_path, output_txt_path):
    """将图像像素数据保存到 TXT 文件中。"""
    print("正在提取像素数据...")
    pixel_data = get_pixel_data(image_path)
    if not pixel_data:
        return

    try:
        print(f"正在保存数据到 '{output_txt_path}'...")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for line in pixel_data:
                f.write(line + "\n")
        print("✅ 成功保存到 TXT 文件！")
    except Exception as e:
        print(f"保存到 TXT 时发生错误：{e}")


# 鼠标点击回调函数
def get_rgb_on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pil_img = param["pil_img"]
        r, g, b = pil_img.getpixel((x, y))
        print(f"点击坐标：({x}, {y})")
        print(f"RGB值：({r}, {g}, {b})")


# 主函数：显示图像，并提取像素数据
def main(image_path, output_txt_path):
    # 保存像素数据到 TXT
    save_to_txt(image_path, output_txt_path)
    # 用 PIL 打开 WebP
    pil_img = Image.open(image_path).convert("RGB")
    # 转换为 OpenCV 格式
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 设置窗口
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)   # 窗口可调节大小
    cv2.resizeWindow('Image', 1200, 800)          # 设置合适的初始大小

    # 绑定鼠标点击事件
    cv2.setMouseCallback('Image', get_rgb_on_click, {"pil_img": pil_img})

    # 显示图像
    while True:
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# 示例调用
image_path = "C:\\Users\\28162\\Desktop\\中石油课题\\数据集\\数据测试图片\\zihong.webp"  # 替换为你的WEBP图像路径
output_txt_path = "pixel_data.txt"  # 保存TXT文件路径
main(image_path, output_txt_path)
