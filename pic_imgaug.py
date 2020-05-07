import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import glob
from imgaug import augmenters as iaa # 引入数据增强的包
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image

sometimes = lambda aug: iaa.Sometimes(0.5, aug) # 建立lambda表达式，
path_re = './paxing_800/'

def load_batch():
    # 加载图片批次
    # Return a numpy array of shape (N, height, width, #channels)
    # or a list of (height, width, #channels) arrays (may have different image sizes).
    # Images should be in RGB for colorspace augmentations.
    # (cv2.imread() returns BGR!)
    # Images should usually be in uint8 with values from 0-255.
    cate = [path_re + x for x in os.listdir(path_re) if os.path.isdir(path_re + x)]
    print(cate)
    img_all = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            try:
                img = cv2.imread(im)
                img_all.append(img[...,::-1])
            except:
                print(im)
                shutil.move(im, '/home/ugrad/Shang/animal/test废/1')
        images_aug = seq.augment_images(images=img_all)  # done by the library
        train_on_images(images_aug, folder + '/')
        img_all=[]

# 行列
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col

# plt绘制
def train_on_images(images,path,iter):
    plt.figure()
    row, col = get_row_col(len(images))
    for i in range(0, len(images)):
        plt.subplot(row, col, i + 1)
        plt.title('Loop3')
        plt.imshow(images[i])
        plt.axis('off')
    #plt.savefig('D:/me/animal/test/feature_map1.png')  # 保存图像到本地
    plt.show()
    '''
    for k,i in zip(range(len(images)),images):
        # i=cv2.resize(i,(224,224))
        cv2.imwrite(path+str(k)+'_'+str(iter)+'.jpg',i[:,:,::-1])'''


# Pipeline:
# (1) Crop images from each side by 1-16px, do not resize the results
#     images back to the input size. Keep them at the cropped size.
# (2) Horizontally flip 50% of the images.
# (3) Blur images using a gaussian kernel with sigma between 0.0 and 3.0.
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # 对50%的图像进行镜像翻转
        iaa.Flipud(0.2),  # 对20%的图像做左右翻转

        sometimes(iaa.Crop(percent=(0, 0.1))),
        # 这里沿袭我们上面提到的sometimes，对随机的一部分图像做crop操作
        # crop的幅度为0到10%

        sometimes(iaa.Affine(  # 对一部分图像做仿射变换
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
            rotate=(-45, 45),  # 旋转±45度之间
            shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
            order=[0, 1],  # 使用最邻近差值或者双线性差值
            cval=(0, 255),  # 全白全黑填充
            mode=ia.ALL  # 定义填充图像外区域的方法
        )),

        # 使用下面的0个到5个之间的方法去增强图像。注意SomeOf的用法
        iaa.SomeOf((0, 5),
                   [
                       # 将部分图像进行超像素的表示。o(╥﹏╥)o用超像素增强作者还是第一次见，比较孤陋寡闻
                       sometimes(
                           iaa.Superpixels(
                               p_replace=(0, 1.0),
                               n_segments=(20, 200)
                           )
                       ),

                       # 用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),
                           iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                           iaa.MedianBlur(k=(3, 11)),
                       ]),

                       # 锐化处理
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                       # 浮雕效果
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                       # 边缘检测，将检测到的赋值0或者255然后叠在原图上
                       sometimes(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0, 0.7)),
                           iaa.DirectedEdgeDetect(
                               alpha=(0, 0.7), direction=(0.0, 1.0)
                           ),
                       ])),

                       # 加入高斯噪声
                       iaa.AdditiveGaussianNoise(
                           loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                       ),

                       # 将1%到10%的像素设置为黑色
                       # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),
                           iaa.CoarseDropout(
                               (0.03, 0.15), size_percent=(0.02, 0.05),
                               per_channel=0.2
                           ),
                       ]),

                       # 5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
                       iaa.Invert(0.05, per_channel=True),

                       # 每个像素随机加减-10到10之间的数
                       iaa.Add((-10, 10), per_channel=0.5),

                       # 像素乘上0.5或者1.5之间的数字.
                       # iaa.Multiply((0.5, 1.5), per_channel=0.5),

                       # 将整个图像的对比度变为原来的一半或者二倍
                       iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.5),

                       # 将RGB变成灰度图然后乘alpha加在原图上
                       iaa.Grayscale(alpha=(0.0, 1.0)),

                       # 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
                       sometimes(
                           iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                       ),

                       # 扭曲图像的局部区域
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                   ],

                   random_order=True  # 随机的顺序把这些操作用在图像上
                   )
    ],
    random_order=True  # 随机的顺序把这些操作用在图像上
)


def resize():
    img_all=[]
    num=0
    cate = [path_re + x for x in os.listdir(path_re) if os.path.isdir(path_re + x)]
    for f in cate:
        print(f)
        for im in glob.glob(f + '/*.jpg'):
            try:
                img2 = Image.open(im)  # Premature end of JPEG file
                img = cv2.imread(im)
                num+=1
                img = cv2.resize(img, (224, 224))
                img_all.append(img)
                cv2.imwrite(im, img)
            except:
                print(im)
                shutil.move(im, './test废')
    print(num)
    img_all=[]

load_batch()
#resize()

