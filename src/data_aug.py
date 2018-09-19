
# coding: utf-8

import shutil
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
from glob import glob
from skimage.io import imread, imsave, imshow


def test_csv(csv_name):
    #本函数生成索引的测试集csv
    name = []
    for root, dirs, files in tqdm(os.walk('./data/test_b/')):
        for file in files:
            if os.path.splitext(file)[-1] == '.jpg':
                name.append(file)

    data = {'name':name}
    data_df = pd.DataFrame(data)
    data_df.to_csv(csv_name, index = None, header = None)

    return name

def cut_img_4(root, filename, x_cut, y_cut, new_fold):
    #切割图片代码，root为来源文件夹，filename为无后缀的文件名，此处传入的filename为无后缀形式
    img = Image.open(os.path.join(root,filename+'.jpg'))
    im0 = img.crop([0, 0, x_cut, y_cut])
    im1 = img.crop([2560-x_cut, 0, 2560, y_cut])
    im2 = img.crop([0, 1920-y_cut, x_cut, 1920])
    im3 = img.crop([2560-x_cut, 1920-y_cut, 2560, 1920])
    im0.save(os.path.join(new_fold,filename+'_0.jpg'))
    im1.save(os.path.join(new_fold,filename+'_1.jpg'))
    im2.save(os.path.join(new_fold,filename+'_2.jpg'))
    im3.save(os.path.join(new_fold,filename+'_3.jpg'))

#生成测试集CSV
name_list = test_csv('./data/test_b.csv')
#裁剪测试集
print('裁剪测试集文件...')
root = './data/test_b'
new_fold = './data/test_b_edit'   #裁剪后的test图片
if os.path.exists(new_fold) == False:
    os.makedirs(new_fold)

n_name_list = len(name_list)-1
x_cut = 1320
y_cut = 1020

new_name = []
for i in tqdm(range(n_name_list)):
    filename = name_list[i]
    tmpname = os.path.splitext(filename)[0]

    cut_img_4(root, tmpname, x_cut, y_cut, new_fold)
    for index in range(4):     #增加照片切割后名称
        new_name.append(tmpname+'_'+str(index)+'.jpg')

new_data = pd.DataFrame({'name':new_name})

new_data.to_csv('./data/test_b_edit.csv', index = None, header = None, columns = ['name'])


#翻转测试集文件
print('翻转测试集文件...')
shutil.copytree("./data/test_b_edit","./data/test_b_edit1")
shutil.copytree("./data/test_b_edit","./data/test_b_edit2")
dirs = ['./data/test_b_edit1', './data/test_b_edit2']
for everydir in dirs:
    filename_list = glob(everydir + '/*.jpg')
    #print(filename_list)
    for filename in filename_list:
        img = cv2.imread(filename)
        if everydir == './data/test_b_edit1':
            img = cv2.flip(img,1,dst=None)
        else:
            img = cv2.flip(img,0,dst=None)
        cv2.imwrite(filename, img)

print('翻转完成！')
