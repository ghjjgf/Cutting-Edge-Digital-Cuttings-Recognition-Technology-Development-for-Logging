import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import matplotlib.pyplot as plt


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
    return (mean_r, mean_g, mean_b), mean_h


def plot_hog(hist):
    """绘制六维HOG柱状图"""
    plt.clf()
    x = np.arange(1, len(hist)+1)
    plt.bar(x, hist, color='blue', alpha=0.7)
    plt.xticks(x, [f"HOG{i}" for i in x])
    plt.ylim(0, 1)
    plt.ylabel("Normalized Magnitude")
    plt.title("Pixel 6-D HOG")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.pause(0.1)


def compute_pixel_hog(img_gray, x, y, bins=6):
    """计算单个像素及其周围3x3邻域的HOG，返回六维HOG均值"""
    h, w = img_gray.shape
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                neighbors.append((nx, ny))

    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle[angle < 0] += 360

    hog_vals = np.zeros(bins, dtype=np.float32)
    for nx, ny in neighbors:
        a = angle[ny, nx]
        m = mag[ny, nx]
        bin_idx = int(a / 360 * bins)
        if bin_idx == bins:
            bin_idx = bins - 1
        hog_vals[bin_idx] += m

    hog_vals /= (hog_vals.sum() + 1e-6)  # 归一化
    return hog_vals


def save_pixel_hog_txt(txt_path, x, y, hog_vals):
    """保存单像素HOG均值到txt"""
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"点击像素点: ({x},{y}) 六维HOG:\n")
        f.write(" ".join([f"{v:.6f}" for v in hog_vals]) + "\n")
        f.write("="*40 + "\n")
    print(f"✅ HOG已保存到 {txt_path}")


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pil_img = param["pil_img"]
        txt_path = param["txt_path"]
        img_gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        rgb_means_list = param["rgb_means_list"]
        h_means_list = param["h_means_list"]

        # 获取RGB和HSV信息
        neighbors, rgb_list, hsv_list = get_rgb_hsv_info(pil_img, x, y)
        rgb_mean, h_mean = save_pixel_info_txt(txt_path, x, y, neighbors, rgb_list, hsv_list)
        rgb_means_list.append(rgb_mean)
        h_means_list.append(h_mean)

        # 计算HOG
        hog_vals = compute_pixel_hog(img_gray, x, y, bins=6)
        save_pixel_hog_txt(txt_path, x, y, hog_vals)
        param["hog_list"].append(hog_vals)

        # 统计并打印结果
        print(f"✅ 已保存RGB、HSV和HOG信息到 '{txt_path}'！")


def main(image_path_str: str, txt_path_str: str):
    image_path = Path(image_path_str)
    txt_path = Path(txt_path_str)

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
    hog_list = []

    # 设置窗口
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 1200, 800)

    # 绑定鼠标点击事件，并传递 txt 路径和均值列表
    cv2.setMouseCallback('Image', on_mouse_click, {
        "pil_img": pil_img,
        "txt_path": txt_path,
        "rgb_means_list": rgb_means_list,
        "h_means_list": h_means_list,
        "hog_list": hog_list
    })

    # 显示图像
    while True:
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    # 输出所有点击像素的HOG和RGB均值统计信息
    if rgb_means_list and h_means_list and hog_list:
        # RGB、H均值统计
        rgb_means_arr = np.array(rgb_means_list)
        h_means_arr = np.array(h_means_list)
        mean_r, mean_g, mean_b = rgb_means_arr.mean(axis=0)
        mean_h = h_means_arr.mean()
        min_r, min_g, min_b = rgb_means_arr.min(axis=0)
        max_r, max_g, max_b = rgb_means_arr.max(axis=0)
        min_h = h_means_arr.min()
        max_h = h_means_arr.max()

        # HOG统计
        hog_arr = np.array(hog_list)
        mean_hog = hog_arr.mean(axis=0)
        min_hog = hog_arr.min(axis=0)
        max_hog = hog_arr.max(axis=0)

        # 写入TXT
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("\n所有点击像素的RGB均值的均值: (%.1f, %.1f, %.1f)\n" % (mean_r, mean_g, mean_b))
            f.write("所有点击像素的H均值的均值: %.1f\n" % mean_h)
            f.write("所有点击像素的RGB均值最小值: (%.0f, %.0f, %.0f)\n" % (min_r, min_g, min_b))
            f.write("所有点击像素的RGB均值最大值: (%.0f, %.0f, %.0f)\n" % (max_r, max_g, max_b))
            f.write("所有点击像素的H均值最小值: %.0f\n" % min_h)
            f.write("所有点击像素的H均值最大值: %.0f\n" % max_h)

            f.write("\n所有点击像素的六维HOG均值:\n")
            f.write(" ".join([f"{v:.6f}" for v in mean_hog]) + "\n")
            f.write("每个维度HOG最小值:\n")
            f.write(" ".join([f"{v:.6f}" for v in min_hog]) + "\n")
            f.write("每个维度HOG最大值:\n")
            f.write(" ".join([f"{v:.6f}" for v in max_hog]) + "\n")
            f.write("="*50 + "\n")

        print("✅ 已输出所有点击像素的RGB、HSV均值和HOG均值")
    else:
        print("未检测到点击，未统计均值。")


if __name__ == '__main__':
    image_path = "H:/dataset/灰白色中砂岩/A/1429.00-灰白色中砂岩-庆阳仪器-白光-小视野.webp"
    clicked_pixels_txt = "C:/Users/28162/Desktop/中石油课题/Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging/click_txtdir/灰白色中砂岩/灰白色中砂岩.txt"
    main(image_path, clicked_pixels_txt)
