import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
import time
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
    if len(before.shape) == 2:
        plt.imshow(before, cmap='gray')
    else:
        plt.imshow(before)
    plt.title(title_left)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if len(after.shape) == 2:
        plt.imshow(after, cmap='gray')
    else:
        plt.imshow(after)
    plt.title(title_right)
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
        erode_iterations=7,
        dilate_iterations=5,
        max_workers=8,
        show=True,
        save_name_suffix=None,
        scale_factor=1.0,
        jpeg_quality=85,
        save_all_images=False
    ):

    img = safe_imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    h, w, _ = img.shape
    
    # 如果指定了缩放因子，则降低分辨率以减少内存占用
    if scale_factor != 1.0 and scale_factor > 0:
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
        print(f"  分辨率缩放: {h}x{w} (缩放因子: {scale_factor})")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------- 1. 去噪与增强 ----------
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_eq = cv2.equalizeHist(gray_blur)

    # ---------- 2. 形态学开闭去噪 ----------
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
    mask = cv2.morphologyEx(gray_eq, cv2.MORPH_OPEN, kernel_clean, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clean, iterations=2)

    # ---------- 3. 岩屑分离：腐蚀 + 膨胀 ----------
    kernel_sep = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (separate_kernel_size, separate_kernel_size))
    eroded = cv2.erode(mask, kernel_sep, iterations=erode_iterations)
    dilated = cv2.dilate(eroded, kernel_sep, iterations=dilate_iterations)
    mask = dilated

    # ---------- 3.5 二值化（必须） ----------
    # 使用 Otsu 或简单阈值确保为 0/255
    try:
        _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        mask_bin = (mask > 0).astype(np.uint8) * 255
    mask_bin = mask_bin.astype(np.uint8)

    # ---------- 4. 连通域分析 ----------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    print(f"✅ 检测到 {max(0, num_labels - 1)} 块连通区域（除背景）")

    os.makedirs(output_folder, exist_ok=True)
    result_white = np.zeros((h, w, 3), dtype=np.uint8)
    result_overlay = img.copy()

    # thread-safe 收集椭圆与直径信息
    ellipses_info = []
    ellipses_lock = threading.Lock()

    # ---------- 5. 椭圆拟合 + 粒径计算（并发收集） ----------
    def process_region(rock_id):
        if rock_id == 0:
            return
        x, y, w_box, h_box, area = stats[rock_id]
        if area < min_pixels:
            return
        submask = (labels[y:y + h_box, x:x + w_box] == rock_id).astype(np.uint8)
        coords_rc = np.column_stack(np.where(submask > 0))  # rows, cols
        if len(coords_rc) < 5:
            return
        # 转为 (x,y)
        coords_xy = np.fliplr(coords_rc).astype(np.float32)  # (col,row) -> (x,y)
        # fitEllipse 要求至少 5 个点
        try:
            ellipse = cv2.fitEllipse(coords_xy)
        except Exception as e:
            # 拟合失败则跳过
            return
        (cx, cy), (MA, ma), angle = ellipse
        ellipse_global = ((x + cx, y + cy), (MA, ma), angle)

        # 粒径计算（按你的规则）
        # MA 是长轴，ma 是短轴，确保 MA >= ma
        if MA < ma:
            MA, ma = ma, MA
        roundness = ma / MA if MA != 0 else 0
        if roundness > 0.7:
            diameter = (MA + ma) / 2
        else:
            diameter = MA

        with ellipses_lock:
            ellipses_info.append({
                'ellipse': ellipse_global,
                'diameter': float(diameter),
                'area': int(area),
                'bbox': (x, y, w_box, h_box)
            })

    # ---------- 6. 多线程拟合 ----------
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for rid in range(1, num_labels):
            futures.append(executor.submit(process_region, rid))
        # 使用 tqdm 跟踪完成度
        for _ in tqdm(as_completed(futures), total=len(futures), desc="拟合椭圆", ncols=100):
            pass

    # ---------- 7. 主线程绘制椭圆并统计 ----------
    diameters = []
    # 绘制顺序在主线程，避免并发写同一张图像
    for info in ellipses_info:
        ellipse_global = info['ellipse']
        diameter = info['diameter']
        diameters.append(diameter)

        # 在白图上填充
        cv2.ellipse(result_white, ellipse_global, (255, 255, 255), -1)
        # 在 overlay 上画较粗轮廓并融合
        overlay = result_overlay.copy()
        thick = max(6, int(min(h, w) / 200))
        cv2.ellipse(overlay, ellipse_global, (255, 255, 255), thick)
        cv2.addWeighted(overlay, 0.6, result_overlay, 0.4, 0, result_overlay)

    # ---------- 8. 输出平均粒径 ----------
    if diameters:
        mean_diameter = np.mean(diameters)
        print(f"🟢 检测到有效岩屑: {len(diameters)}，平均粒径: {mean_diameter:.2f} 像素")
    else:
        print("⚠️ 未检测到有效岩屑（基于 min_pixels 与连通域分析）")

    # ---------- 9. 拼接显示 + 保存 ----------
    combined = np.hstack([img, result_overlay])

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    suffix = save_name_suffix if save_name_suffix else f"_腐蚀{erode_iterations}_膨胀{dilate_iterations}_侵蚀核{erode_kernel_size}_分离核{separate_kernel_size}"
    save_white = os.path.join(output_folder, f"{base_name}{suffix}_filled_white_ellipses.png")
    save_overlay = os.path.join(output_folder, f"{base_name}{suffix}_ellipses_on_original.png")
    save_combined = os.path.join(output_folder, f"{base_name}{suffix}_side_by_side.png")

    # 使用兼容 Unicode 路径的保存方法
    def _save_image_unicode(path, image, quality=85):
        try:
            save_dir = os.path.dirname(path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            if image is None:
                print(f"错误: 图像数据为空 ({path})")
                return False
            if not isinstance(image, np.ndarray):
                print(f"错误: 图像类型不正确 ({type(image)}) for {path}")
                return False

            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

            try:
                ext = os.path.splitext(path)[1].lower() or '.png'
                
                # 根据文件格式选择压缩参数
                if ext in ['.jpg', '.jpeg']:
                    # JPEG 格式使用质量参数压缩
                    success, buf = cv2.imencode(ext, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                elif ext == '.png':
                    # PNG 使用 9 级压缩（最大压缩）
                    success, buf = cv2.imencode(ext, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                else:
                    # 其他格式默认编码
                    success, buf = cv2.imencode(ext, image)
                
                if success:
                    with open(path, 'wb') as f:
                        f.write(buf.tobytes())
                    
                    # 获取文件大小（KB）
                    file_size = os.path.getsize(path) / 1024
                    print(f"✓ 保存成功: {path} (大小: {file_size:.2f} KB)")
                    return True
                else:
                    print(f"cv2.imencode 失败: {path}")
            except Exception as e:
                print(f"保存时出错 ({path}): {e}")
            return False
        except Exception as e:
            print(f"保存图片过程出现未知错误 ({path}):\n{e}")
            return False

    # 根据参数决定保存哪些图像（节省磁盘空间）
    if save_all_images:
        _save_image_unicode(save_white, result_white, quality=jpeg_quality)
        _save_image_unicode(save_overlay, result_overlay, quality=jpeg_quality)
    
    # 总是保存并排对比图（推荐用于评估）
    _save_image_unicode(save_combined, combined, quality=jpeg_quality)
    
    # 及时释放大对象以减少内存占用
    del result_white, result_overlay, combined, img, gray, gray_blur, gray_eq, mask, mask_bin, eroded, dilated, labels, ellipses_info

    # ---------- 10. 可视化最终效果 ----------
    if show:
        # matplotlib expects RGB ordering
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            white_rgb = cv2.cvtColor(result_white, cv2.COLOR_BGR2RGB)
            show_comparison(img_rgb, white_rgb, "原图", "拟合结果(白填充椭圆)", "④ 连通域 + 椭圆拟合最终效果")
        except Exception:
            # 退回到直接显示（安全）
            show_comparison(img, result_white, "原图", "拟合结果(白填充椭圆)", "④ 连通域 + 椭圆拟合最终效果")

    plt.close('all')
    if save_all_images:
        print(f"✅ 已保存:\n  {save_white}\n  {save_overlay}\n  {save_combined}")
    else:
        print(f"✅ 已保存:\n  {save_combined}")
    
    return len(diameters) if diameters else 0

# ------------------------
# 批量处理函数
# ------------------------
def process_folder(
        input_folder,
        output_folder,
        supported_formats=None,
        erode_iter_range=range(1, 16),
        dilate_iter_range=range(1, 16),
        erode_kernel_sizes=(3, 5, 7),
        separate_kernel_sizes=(3, 5, 7),
        min_pixels=300,
        max_workers=8,
        skip_existing=True,
        scale_factor=1.0,
        jpeg_quality=85,
        save_all_images=False
):
    if supported_formats is None:
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    print(f"\n处理输入文件夹: {input_folder}")
    print(f"输出到文件夹: {output_folder}\n")

    for root, dirs, files in os.walk(input_folder):
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in supported_formats]
        if not image_files:
            continue

        rel_dir = os.path.relpath(root, input_folder)
        dest_dir = os.path.join(output_folder, rel_dir) if rel_dir != '.' else output_folder
        os.makedirs(dest_dir, exist_ok=True)

        for fname in tqdm(image_files, desc=f"处理 {rel_dir}", leave=False):
            in_path = os.path.join(root, fname)
            base_name = os.path.splitext(fname)[0]
            print(f"\n开始处理: {fname}")

            for e_iter in erode_iter_range:
                for d_iter in dilate_iter_range:
                    for e_k in erode_kernel_sizes:
                        for s_k in separate_kernel_sizes:
                            suffix = f"_腐蚀{e_iter}_膨胀{d_iter}_侵蚀核{e_k}_分离核{s_k}"
                            out_name = f"{base_name}{suffix}_ellipses_on_original.png"
                            out_path = os.path.join(dest_dir, out_name)
                            if skip_existing and os.path.exists(out_path):
                                print("false")
                                continue
                            try:
                                print("success")
                                segment_and_fit_ellipses(
                                    in_path,
                                    dest_dir,
                                    min_pixels=min_pixels,
                                    erode_kernel_size=e_k,
                                    separate_kernel_size=s_k,
                                    erode_iterations=e_iter,
                                    dilate_iterations=d_iter,
                                    max_workers=max_workers,
                                    show=False,
                                    save_name_suffix=suffix,
                                    scale_factor=scale_factor,
                                    jpeg_quality=jpeg_quality,
                                    save_all_images=save_all_images
                                )
                            except Exception as ex:
                                print(f"处理 {in_path} 时出错（参数 {suffix}）: {ex}")

    print(f"批量处理完成，结果保存在: {output_folder}")

# ------------------------
# 主程序入口（示例）
# ------------------------
if __name__ == "__main__":
    # 单张图像调用示例（取消注释并修改路径来测试单图）
    # segment_and_fit_ellipses(
    #     r"path/to/your/image.png",
    #     output_folder=r"path/to/output/folder",
    #     min_pixels=300,
    #     erode_kernel_size=3,
    #     separate_kernel_size=9,
    #     max_workers=8,
    #     show=True
    # )

    # 批量处理示例（修改为你的路径）
    t1 = time.time()
    process_folder(
        input_folder = r"E:/课题/中石油课题/粒径进展/测试集/中砂岩/子图",
        output_folder = r"E:/课题/中石油课题/粒径进展/测试集/中砂岩/粒径结果",
        supported_formats=None,
        erode_iter_range=range(1, 15),   # 示例：把范围缩小以便快速测试
        dilate_iter_range=range(1, 15),
        erode_kernel_sizes=(3, 5),
        separate_kernel_sizes=(3, 5),
        min_pixels=300,
        max_workers=16,
        skip_existing=True,
        scale_factor=0.5,                 # 缩放到 50% 以减少内存占用
        jpeg_quality=85,                  # JPEG 质量 (1-100, 越低越小)
        save_all_images=False             # 仅保存对比图，不保存单个结果图
    )
    t2 = time.time()
    print(f"总处理时间: {t2 - t1:.2f} 秒")
