import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import glob
from imgaug import augmenters as iaa #引入数据增强的包
import matplotlib.pyplot as plt



def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def train_on_images(images):
    plt.figure()
    row, col = get_row_col(len(images))
    for i in range(0, len(images)):
        plt.subplot(row, col, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
    #plt.savefig('D:/me/animal/test/feature_map1.png')  # 保存图像到本地
    plt.show()
    '''
        for k,i in zip(range(len(images)),images):
            # i=cv2.resize(i,(224,224))
            cv2.imwrite(path+str(k)+'_'+str(iter)+'.jpg',i[:,:,::-1])'''


seq = iaa.Sequential([
    #iaa.PiecewiseAffine(scale=(0.01, 0.05)),
    #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05),per_channel=0.2),
    #iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
    #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
    #iaa.Superpixels(p_replace=(0, 1.0),n_segments=(20, 200)),
    #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    iaa.Crop(px=(1, 16), keep_size=False),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0))
])

img=[]
img_all = []
img1 = cv2.imread('./13_.jpg')
img.append(img1[...,::-1])
images_aug = seq.augment_images(images=img)  # done by the library
train_on_images(images_aug)