import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import csv


# 将图像像素数据保存到 CSV 文件
def save_to_csv(image_path: Path, output_csv_path: Path):
    """将图像所有像素数据（x, y, r, g, b）保存到 CSV 文件中。"""
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        pixels = img.load()

        print(f"正在保存数据到 '{output_csv_path}'...")

        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "r", "g", "b"])
            for y in range(height):
                for x in range(width):
                    r, g, b = pixels[x, y]
                    writer.writerow([x, y, r, g, b])

        print("✅ 成功保存到 CSV 文件！")
    except FileNotFoundError:
        print(f"错误：找不到文件 '{image_path}'")
        return None
    except Exception as e:
        print(f"保存到 CSV 时发生错误：{e}")
        return None


# 鼠标点击回调函数
def get_rgb_on_click(event, x, y, flags, param):
    """鼠标点击回调函数，获取 RGB 值并保存到 CSV。"""
    if event == cv2.EVENT_LBUTTONDOWN:
        pil_img = param["pil_img"]
        output_csv_path = param["click_data_path"]

        # 确保坐标在图像范围内
        if 0 <= x < pil_img.width and 0 <= y < pil_img.height:
            r, g, b = pil_img.getpixel((x, y))
            print(f"点击坐标：({x}, {y})")
            print(f"RGB值：({r}, {g}, {b})")
            # 将点击数据追加保存到 CSV 文件
            try:
                with open(output_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(["x", "y", "r", "g", "b"])
                    writer.writerow([x, y, r, g, b])
                print(f"✅ 点击数据成功保存到 '{output_csv_path}'！")
            except Exception as e:
                print(f"保存点击数据时发生错误：{e}")


# 主函数：显示图像，并提取像素数据
def main(image_path_str: str, all_pixels_csv_path: str, click_data_csv_path: str):
    """
    主函数：显示图像，并提取像素数据。
    """
    image_path = Path(image_path_str)
    all_pixels_csv_path = Path(all_pixels_csv_path)
    click_data_csv_path = Path(click_data_csv_path)

    # 检查文件是否存在
    if not image_path.exists():
        print(f"错误：找不到图像文件 '{image_path}'。请检查路径。")
        sys.exit(1)

    # 保存所有像素数据到 CSV
    save_to_csv(image_path, all_pixels_csv_path)

    # 用 PIL 打开图像
    pil_img = Image.open(image_path).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 设置窗口
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 1200, 800)

    # 绑定鼠标点击事件，并传递 click_data_csv_path
    cv2.setMouseCallback('Image', get_rgb_on_click, {"pil_img": pil_img, "click_data_path": click_data_csv_path})

    # 显示图像
    while True:
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    # 计算平均 RGB 值并保存到 CSV
    calculate_and_save_average_rgb(click_data_csv_path)


def calculate_and_save_average_rgb(file_path):
    """
    从 CSV 文件中读取 RGB 值，计算平均值，并追加到文件末尾。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"\n警告：未找到点击数据文件 '{file_path}'，跳过平均值计算。")
        return

    print("\n正在计算点击像素的平均 RGB 值...")

    rgb_values = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # 跳过表头
            next(reader)
            for row in reader:
                if len(row) >= 5:
                    r, g, b = int(row[2]), int(row[3]), int(row[4])
                    rgb_values.append([r, g, b])
    except Exception as e:
        print(f"读取点击数据时发生错误：{e}")
        return

    if not rgb_values:
        print("没有点击数据，无法计算平均值。")
        return

    # 使用 NumPy 数组计算平均值
    avg_rgb = np.mean(rgb_values, axis=0)
    avg_r, avg_g, avg_b = int(round(avg_rgb[0])), int(round(avg_rgb[1])), int(round(avg_rgb[2]))

    print(f"所有点击像素的平均 RGB 值为: ({avg_r}, {avg_g}, {avg_b})")

    # 将平均值追加写入 CSV 文件
    try:
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Average", "N/A", avg_r, avg_g, avg_b])
        print(f"✅ 平均 RGB 值已成功保存到 '{file_path}'！")
    except Exception as e:
        print(f"保存平均值时发生错误：{e}")


if __name__ == '__main__':
    image_path = "dataset/train/灰白色/1092.00-灰白色中砂岩-庆阳仪器-白光-小视野.webp"
    all_pixels_output = "all_pixel_data.csv"
    clicked_pixels_output = "clicked_pixel_data.csv"
    main(image_path, all_pixels_output, clicked_pixels_output)
