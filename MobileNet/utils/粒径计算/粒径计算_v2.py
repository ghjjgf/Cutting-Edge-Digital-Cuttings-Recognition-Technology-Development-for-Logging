import cv2 
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ------------------------
# 安全读取图片
# ------------------------
def safe_imread(path):
    try:
        img_pil = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"❌ 无法读取图片: {path}\n错误信息: {e}")
        return None

# ------------------------
# 可视化对比显示函数
# ------------------------
def show_comparison(before, after, title_left, title_right, step_name):
    plt.figure(figsize=(8, 4))
    plt.suptitle(step_name, fontsize=14)

    plt.subplot(1, 2, 1)
    plt.imshow(before, cmap='gray' if len(before.shape) == 2 else None)
    plt.title(title_left)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(after, cmap='gray' if len(after.shape) == 2 else None)
    plt.title(title_right)
    plt.axis('off')

    plt.tight_layout()
    plt.show(block=True)  # 等用户关闭窗口后继续

# ------------------------
# 岩屑分割 + 椭圆拟合 + 粒径计算
# ------------------------
def segment_and_fit_ellipses(
        image_path,
        output_folder,
        min_pixels=300,
        erode_kernel_size=3,
        separate_kernel_size=5,
        max_workers=8
        ):

    img = safe_imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------- 1. 去噪与增强 ----------
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    # ---------- 2. 形态学开闭去噪 ----------
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
    mask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_clean, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)

    # ---------- 3. 岩屑分离：腐蚀 + 膨胀 ----------
    kernel_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (separate_kernel_size, separate_kernel_size))
    eroded = cv2.erode(mask, kernel_sep, iterations=7)
    dilated = cv2.dilate(eroded, kernel_sep, iterations=5)
    mask = dilated

    # ---------- 4. 连通域分析 ----------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    print(f"✅ 检测到 {num_labels - 1} 块岩屑（除背景）")

    os.makedirs(output_folder, exist_ok=True)
    result_white = np.zeros((h, w, 3), dtype=np.uint8)
    result_overlay = img.copy()
    diameters = []  # 存储每个岩屑粒径

    # ---------- 5. 椭圆拟合 + 粒径计算 ----------
    def process_region(rock_id):
        if rock_id == 0:
            return
        x, y, w_box, h_box, area = stats[rock_id]
        if area < min_pixels:
            return
        submask = (labels[y:y + h_box, x:x + w_box] == rock_id).astype(np.uint8)
        coords = np.column_stack(np.where(submask > 0))
        if len(coords) < 5:
            return
        coords = np.fliplr(coords)  # (row,col) -> (x,y)
        ellipse = cv2.fitEllipse(coords)
        (cx, cy), (MA, ma), angle = ellipse
        ellipse_global = ((x + cx, y + cy), (MA, ma), angle)

        # 绘制椭圆
        cv2.ellipse(result_white, ellipse_global, (255, 255, 255), -1)
        overlay = result_overlay.copy()
        cv2.ellipse(overlay, ellipse_global, (0, 0, 255), 2)
        cv2.addWeighted(overlay, 0.5, result_overlay, 0.5, 0, result_overlay)

        # 粒径计算
        roundness = ma / MA
        if roundness > 0.7:
            diameter = (MA + ma) / 2
        else:
            diameter = MA
        diameters.append(diameter)

    # ---------- 6. 多线程拟合 ----------
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_region, rid) for rid in range(1, num_labels)]
        for _ in tqdm(as_completed(futures), total=num_labels - 1, desc="拟合椭圆", ncols=100):
            pass

    # ---------- 7. 输出平均粒径 ----------
    if diameters:
        mean_diameter = np.mean(diameters)
        print(f"🟢 平均粒径: {mean_diameter:.2f} 像素")
    else:
        print("⚠️ 未检测到有效岩屑")

    # ---------- 8. 拼接显示 + 保存 ----------
    combined = np.hstack([img, result_overlay])
    save_white = os.path.join(output_folder, "filled_white_ellipses.png")
    save_overlay = os.path.join(output_folder, "ellipses_on_original.png")
    save_combined = os.path.join(output_folder, "side_by_side.png")

    cv2.imencode(".png", result_white)[1].tofile(save_white)
    cv2.imencode(".png", result_overlay)[1].tofile(save_overlay)
    cv2.imencode(".png", combined)[1].tofile(save_combined)

    # ---------- 9. 可视化最终效果 ----------
    show_comparison(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(result_white, cv2.COLOR_BGR2RGB),
                    "原图", "拟合结果", "④ 连通域 + 椭圆拟合最终效果")

    print(f"✅ 已保存:\n  {save_white}\n  {save_overlay}\n  {save_combined}")
    plt.close('all')
    return result_overlay

# ------------------------
# 主程序入口
# ------------------------
if __name__ == "__main__":
    segment_and_fit_ellipses(
        r"E:\课题\中石油课题\Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging\MobileNet\utils\粒径保存\1272.00-灰白色中砂岩-庆阳仪器-白光-小视野.webp_segmented_2(1).png",
        output_folder=r"E:\课题\中石油课题\Cutting-Edge-Digital-Cuttings-Recognition-Technology-Development-for-Logging\MobileNet\utils\粒径保存\优化后",
        min_pixels=300,
        erode_kernel_size=3,
        separate_kernel_size=9,
        max_workers=32
    )
