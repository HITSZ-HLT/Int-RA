import cv2,os
import numpy as np
import matplotlib.pyplot as plt

def cal_attention(img_file):
    # 读取图片
    image = cv2.imread(img_file)
    print(os.path.exists(img_file))
    # print(image)
    # 计算空间注意力，这里简单地使用图像的红色通道作为注意力权重
    attention_map = image[:, :, 2] / 255.0  # 使用红色通道作为注意力权重，范围从0到1

    # 将注意力图进行平滑处理，以增强可视化效果
    attention_map = cv2.GaussianBlur(attention_map, (21, 21), 11)

    # 将图像和注意力图结合起来，形成热图
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)

    # 将热图叠加在原始图像上
    result = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    # 显示原始图像、注意力图和叠加后的结果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(attention_map, cmap='hot')
    plt.title('Attention Map')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Heatmap Overlay')
    plt.axis('off')
    file_name = img_file.split('/')[-1]
    plt.savefig('out_1/'+file_name)

cal_attention('boba_342.jpg')
# root = '/data10T/wangbingbing/Chinese-CLIP/Dataset/total/total_img_2'
# length = len(os.listdir(root))
# n = 1
# for file in os.listdir(root):
#     print(n,length)
#     print(os.path.join(root,file))
#     cal_attention(os.path.join(root,file))
#     n+=1