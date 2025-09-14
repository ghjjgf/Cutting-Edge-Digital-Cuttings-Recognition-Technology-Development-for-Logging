from PIL import Image

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


# 示例调用
save_to_txt(
    "C:\\Users\\28162\\Desktop\\中石油课题\\数据集\\数据测试图片\\zihong.webp",
    "pixel_data.txt"
)
