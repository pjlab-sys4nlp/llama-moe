from PIL import Image


def save_images_as_gif(image_paths, output_path, duration=200):
    """
    将图像文件路径列表保存为 GIF 动画文件。

    :param image_paths: 包含图像文件路径的列表
    :param output_path: 保存 GIF 动画的文件路径
    :param duration: 每帧之间的时间间隔（毫秒）
    """
    if not image_paths:
        print("Error: No image paths provided.")
        return

    try:
        # 打开图像文件并将它们添加到图像列表中
        images = [Image.open(image_path) for image_path in image_paths]

        # 保存 GIF 动画
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=duration,
        )
        print(f"GIF animation saved as {output_path}")
    except Exception as e:
        print(f"Error: {e}")


# 使用示例
image_paths = ["image1.png", "image2.png", "image3.png"]
output_gif = "output.gif"

save_images_as_gif(image_paths, output_gif)
