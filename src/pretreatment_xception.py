# coding: utf-8
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ## pretreatment信息模型预测
width = height = 512
import keras
import keras.callbacks
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, rmsprop, SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception,preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input

base_model = Xception(weights=None,
                      input_shape = (width,height,3),
                      include_top=False)
model_out = base_model.output
avg = GlobalAveragePooling2D()(model_out)
dense = Dense(128, activation='relu')(avg)
predictions = Dense(2, activation='softmax')(dense)
model_test = Model(inputs = base_model.input, outputs = predictions)

model_test.load_weights('./models/pretreatment_models/xception_0731_edit.h5')


# ## 预测
def label_combine(prob):
    new_prob = 1
    for i in range(4): 
        prob[i] = 1 - prob[i]
        new_prob *= prob[i]
    
    new_prob = 1 - new_prob
    return new_prob

def build_csv(filenames, test_np, csv_name):
    print(test_np.shape)
    print(test_np)
    probability = []
    # 处理预测结果，使其在（0,1）中
    for i in range(len(test_np)):
        p = test_np[i][1]    #当瑕疵为0时，参数为0；瑕疵为1时，参数为1
        probability.append(p)
    new_filename = []
    new_probability = []
    for i in range(0,len(filenames),4):
        name_list = filenames[i].split('_')
        n_name_list = len(name_list) - 2
        # 生成名称
        filename = ''
        for j in range(n_name_list): 
            filename = filename + str(name_list[j]) + '_'
        filename = filename+ name_list[-2]+ '.jpg'
        new_filename.append(filename)

        pro_list = [ float(probability[i+index])  for index in range(4) ]
        new_probability.append(label_combine(pro_list)) 

    df_data = pd.DataFrame({'probability':new_probability, 'filename':new_filename})
    df_data.to_csv( csv_name, index = False, columns = ['filename','probability'])

#####读取测试集！！！
df_test = pd.read_csv('./data/test_b_edit.csv', header = None)
df_test.columns = ['filename']
filenames = df_test['filename']
df_test.reset_index(inplace=True)
n_test = len(df_test)
print('测试集数据个数: {0}'.format(n_test))
# 预测
print('Start pretreatment predict...')
X_test = np.zeros((n_test, width, height, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    img = cv2.imread('./data/test_b_edit/{0}'.format(df_test['filename'][i]))
    X_test[i] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA )
test_np = model_test.predict(X_test, batch_size=75, verbose=1)
build_csv(filenames, test_np, './results_tmp/pretreatment_3.csv')
# 测试增强1
X_test = np.zeros((n_test, width, height, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    img = cv2.imread('./data/test_b_edit1/{0}'.format(df_test['filename'][i]))
    X_test[i] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA )
test_np = model_test.predict(X_test, batch_size=75, verbose=1)
build_csv(filenames, test_np, './results_tmp/pretreatment_4.csv')
# 测试增强2
X_test = np.zeros((n_test, width, height, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    img = cv2.imread('./data/test_b_edit2/{0}'.format(df_test['filename'][i]))
    X_test[i] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA )
test_np = model_test.predict(X_test, batch_size=75, verbose=1)
build_csv(filenames, test_np, './results_tmp/pretreatment_5.csv')

