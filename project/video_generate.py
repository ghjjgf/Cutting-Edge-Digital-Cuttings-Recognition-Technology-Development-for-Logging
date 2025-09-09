from moviepy import ImageSequenceClip
import os

image_folder = "images_for_video"
image_files = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".webp")])

if not image_files:
    print("在指定的文件夹中未找到任何图片。")
else:
    # 设置每张图片的停留时间（秒）
    duration_per_image = 2

    # 创建一个列表，指定每张图片的持续时间
    durations = [duration_per_image] * len(image_files)

    # 使用 ImageSequenceClip 将图片序列合成为视频剪辑
    clip = ImageSequenceClip(image_files, durations=durations)

    # 设置输出视频的文件名
    output_video_file = "10_second_video.mp4"

    # 将视频剪辑写入文件
    clip.write_videofile(output_video_file, fps=2)  # fps = 2 0.5秒检测一次

    print(f"\n视频已成功创建：{output_video_file}")
