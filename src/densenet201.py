# coding: utf-8

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from sklearn.utils import shuffle 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ## 搭建网络
width = height = 512
import keras
import keras.callbacks
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam, rmsprop, SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.xception import Xception,preprocess_input
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.densenet import DenseNet201, preprocess_input

base_model = DenseNet201(weights=None,
                      input_shape = (width,height,3),
                      include_top=False)
model_out = base_model.output
avg = GlobalAveragePooling2D()(model_out)
dense = Dense(256, activation='relu')(avg)
dense = Dropout(0.5)(dense)
predictions = Dense(11, activation='softmax')(dense)
model_test = Model(inputs = base_model.input, outputs = predictions)


# ## 预测
def build_csv(df_test, test_np, csv_name):
    print(test_np.shape)#加油
    all_prob = []
    for i in range(n_test):
        prob = str(test_np[i]).split('[')[1].split(']')[0].split(' ')
        for i in range(len(test_np[0])):
            if '\n' in prob[i]:
                prob[i] = prob[i][:-1]
        prob = [float(x) for x in prob if x != '']        
        all_prob.append(prob)
        #all_prob[i] += test_np[i]

    df_test['probability'] = all_prob
    df_test.to_csv(csv_name, index=False, columns = ['filename', 'probability'])
    
if os.path.exists('./results_tmp') == False:
    os.makedirs('./results_tmp')
    
# 读取测试集！！！
df_test = pd.read_csv('./data/test_b_edit.csv', header = None)
df_test.columns = ['filename']
df_test.reset_index(inplace=True)
n_test = len(df_test)

print('测试集数据个数: {0}'.format(n_test))
# 测试
print('Start predict...')
X_test = np.zeros((n_test, width, height, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    img = cv2.imread('./data/test_b_edit/{0}'.format(df_test['filename'][i]))
    X_test[i] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA )

model_test.load_weights('./models/densenet201_0821_256fc_last_epoch.h5')    
test_np = model_test.predict(X_test, batch_size=100, verbose = 1)
build_csv(df_test,test_np,'./results_tmp/densenet201_0.csv')
model_test.load_weights('./models/densenet201_0824_256fc_last_epoch.h5')    
test_np = model_test.predict(X_test, batch_size=100, verbose = 1)
build_csv(df_test,test_np,'./results_tmp/densenet201_1.csv')

# 测试增强1
X_test = np.zeros((n_test, width, height, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    img = cv2.imread('./data/test_b_edit1/{0}'.format(df_test['filename'][i]))
    X_test[i] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA )
    
model_test.load_weights('./models/densenet201_0821_256fc_last_epoch.h5')    
test_np = model_test.predict(X_test, batch_size=100, verbose = 1)
build_csv(df_test,test_np,'./results_tmp/densenet201_2.csv')

model_test.load_weights('./models/densenet201_0824_256fc_last_epoch.h5')    
test_np = model_test.predict(X_test, batch_size=100, verbose = 1)
build_csv(df_test,test_np,'./results_tmp/densenet201_3.csv')

# 测试增强2
X_test = np.zeros((n_test, width, height, 3), dtype=np.uint8)
for i in tqdm(range(n_test)):
    img = cv2.imread('./data/test_b_edit2/{0}'.format(df_test['filename'][i]))
    X_test[i] = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA )
    
model_test.load_weights('./models/densenet201_0821_256fc_last_epoch.h5')    
test_np = model_test.predict(X_test, batch_size=100, verbose = 1)
build_csv(df_test,test_np,'./results_tmp/densenet201_4.csv')

model_test.load_weights('./models/densenet201_0824_256fc_last_epoch.h5')    
test_np = model_test.predict(X_test, batch_size=100, verbose = 1)
build_csv(df_test,test_np,'./results_tmp/densenet201_5.csv')


