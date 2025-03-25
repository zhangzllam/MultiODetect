from PIL import Image, ImageChops

def are_images_identical(image1_path, image2_path):
    # 打开两张图片
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # 如果图片尺寸不一致，则一定不同
    if img1.size != img2.size:
        print("!")
        return False

    # 计算两张图片的差异
    diff = ImageChops.difference(img1, img2)

    # 判断差异图像是否为纯黑
    if diff.getbbox() is None:
        return True
    else:
        return False

# 示例用法
image1_path = r"C:\Users\zhang\Desktop\tmp\3_60_000.bmp"
image2_path = r"C:\Users\zhang\Desktop\tmp\tmp.bmp"
are_identical = are_images_identical(image1_path, image2_path)

print("两张图片完全相同" if are_identical else "两张图片不同")
