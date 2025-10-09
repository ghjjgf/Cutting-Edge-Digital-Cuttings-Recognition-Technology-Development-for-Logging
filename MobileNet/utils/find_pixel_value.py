import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys


def get_neighbors(x, y, width, height):
    """获取(x, y)及其周围8个像素的坐标"""
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
    return neighbors


def get_rgb_hsv_info(pil_img, x, y):
    """获取(x, y)及其周围8个像素的RGB和HSV(H)信息"""
    width, height = pil_img.size
    neighbors = get_neighbors(x, y, width, height)
    rgb_list = []
    hsv_list = []
    img_np = np.array(pil_img)
    img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    for nx, ny in neighbors:
        r, g, b = pil_img.getpixel((nx, ny))
        h = img_hsv[ny, nx, 0]  # 注意顺序：y, x
        rgb_list.append((r, g, b))
        hsv_list.append(h)
    return neighbors, rgb_list, hsv_list


def save_pixel_info_txt(txt_path, x, y, neighbors, rgb_list, hsv_list):
    """美观格式输出并保存到txt"""
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"点击像素点: ({x}, {y})\n")
        f.write("像素点及其周围8个点的RGB值和HSV(H)值：\n")
        f.write(f"{'坐标':^12}{'R':^6}{'G':^6}{'B':^6}{'H':^6}\n")
        for (nx, ny), (r, g, b), h in zip(neighbors, rgb_list, hsv_list):
            f.write(f"({nx:>4},{ny:<4}) {r:>4}  {g:>4}  {b:>4}  {h:>4}\n")
        # 计算均值
        rgb_arr = np.array(rgb_list)
        hsv_arr = np.array(hsv_list)
        mean_r, mean_g, mean_b = rgb_arr.mean(axis=0)
        mean_h = hsv_arr.mean()
        min_r, min_g, min_b = rgb_arr.min(axis=0)
        max_r, max_g, max_b = rgb_arr.max(axis=0)
        min_h = hsv_arr.min()
        max_h = hsv_arr.max()
        f.write("\nRGB均值: (%.1f, %.1f, %.1f)\n" % (mean_r, mean_g, mean_b))
        f.write("H均值: %.1f\n" % mean_h)
        f.write("RGB最小值: (%.0f, %.0f, %.0f)\n" % (min_r, min_g, min_b))
        f.write("RGB最大值: (%.0f, %.0f, %.0f)\n" % (max_r, max_g, max_b))
        f.write("H最小值: %.0f\n" % min_h)
        f.write("H最大值: %.0f\n" % max_h)
        f.write("="*40 + "\n")
    # 返回均值，便于后续统计
    return (mean_r, mean_g, mean_b), mean_h


# 鼠标点击回调函数
def get_rgb_on_click(event, x, y, flags, param):
    """鼠标点击回调函数，获取 RGB 和 HSV 信息并保存到 TXT。"""
    if event == cv2.EVENT_LBUTTONDOWN:
        pil_img = param["pil_img"]
        output_txt_path = param["click_data_txt"]
        rgb_means_list = param["rgb_means_list"]
        h_means_list = param["h_means_list"]

        # 确保坐标在图像范围内
        if 0 <= x < pil_img.width and 0 <= y < pil_img.height:
            print(f"点击坐标：({x}, {y})")
            neighbors, rgb_list, hsv_list = get_rgb_hsv_info(pil_img, x, y)
            rgb_mean, h_mean = save_pixel_info_txt(output_txt_path, x, y, neighbors, rgb_list, hsv_list)
            rgb_means_list.append(rgb_mean)
            h_means_list.append(h_mean)
            print(f"✅ 邻域RGB和H信息已保存到 '{output_txt_path}'！")


# 主函数：显示图像，点击并保存信息到TXT，退出后统计均值
def main(image_path_str: str, click_data_txt_path: str):
    image_path = Path(image_path_str)
    click_data_txt_path = Path(click_data_txt_path)

    # 检查文件是否存在
    if not image_path.exists():
        print(f"错误：找不到图像文件 '{image_path}'。请检查路径。")
        sys.exit(1)

    # 用 PIL 打开图像
    pil_img = Image.open(image_path).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 用于存储所有点击的均值
    rgb_means_list = []
    h_means_list = []

    # 设置窗口
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 1200, 800)

    # 绑定鼠标点击事件，并传递 txt 路径和均值列表
    cv2.setMouseCallback('Image', get_rgb_on_click, {
        "pil_img": pil_img,
        "click_data_txt": click_data_txt_path,
        "rgb_means_list": rgb_means_list,
        "h_means_list": h_means_list
    })

    # 显示图像
    while True:
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    # 统计所有点击的均值的均值 
    if rgb_means_list and h_means_list:
        rgb_means_arr = np.array(rgb_means_list)
        h_means_arr = np.array(h_means_list)
        mean_r, mean_g, mean_b = rgb_means_arr.mean(axis=0)
        mean_h = h_means_arr.mean()
        min_r, min_g, min_b = rgb_means_arr.min(axis=0)
        max_r, max_g, max_b = rgb_means_arr.max(axis=0)
        min_h = h_means_arr.min()
        max_h = h_means_arr.max()
        # 统计所有点击的HOG值
        
        print("\n所有点击像素的RGB均值的均值: (%.1f, %.1f, %.1f)" % (mean_r, mean_g, mean_b))
        print("所有点击像素的H均值的均值: %.1f" % mean_h)
        print("所有点击像素的RGB均值最小值: (%.0f, %.0f, %.0f)" % (min_r, min_g, min_b))
        print("所有点击像素的RGB均值最大值: (%.0f, %.0f, %.0f)" % (max_r, max_g, max_b))
        print("所有点击像素的H均值最小值: %.0f" % min_h)
        print("所有点击像素的H均值最大值: %.0f" % max_h)
        print(f"一共点击了 {len(rgb_means_list)} 个像素点。")
        # 追加写入到txt
        with open(click_data_txt_path, "a", encoding="utf-8") as f:
            f.write("\n所有点击像素的RGB均值的均值: (%.1f, %.1f, %.1f)\n" % (mean_r, mean_g, mean_b))
            f.write("所有点击像素的H均值的均值: %.1f\n" % mean_h)
            f.write("所有点击像素的RGB均值最小值: (%.0f, %.0f, %.0f)\n" % (min_r, min_g, min_b))
            f.write("所有点击像素的RGB均值最大值: (%.0f, %.0f, %.0f)\n" % (max_r, max_g, max_b))
            f.write("所有点击像素的H均值最小值: %.0f\n" % min_h)
            f.write("所有点击像素的H均值最大值: %.0f\n" % max_h)
            f.write("="*40 + "\n")
            # 一共有几个点
            f.write(f"一共点击了 {len(rgb_means_list)} 个像素点。\n")
    else:
        print("未检测到点击，未统计均值。")


if __name__ == '__main__':
    image_path = "H:/dataset/紫红色泥岩/A/Purple-red-mudstone_33.webp"
    clicked_pixels_txt = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/click_txtdir/Purple-red-mudstone32.txt"
    main(image_path, clicked_pixels_txt)
