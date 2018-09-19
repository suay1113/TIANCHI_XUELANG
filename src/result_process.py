
# coding: utf-8

# In[1]:


import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

sub_label_list = ['norm', 'defect_1', 'defect_2', 'defect_3', 'defect_4', 'defect_5',
                  'defect_6', 'defect_7', 'defect_8', 'defect_9', 'defect_10']
def get_max(prob):
    max_index = np.zeros(2)
    max_value = 0
    for i in range(len(prob)):
        for j in range(1, len(prob[0])):
            if prob[i][j] > max_value:
                max_value = prob[i][j]
                max_index = [i, j]
                
    return max_value, max_index

def normalization(L):
    return [a/np.sum(L) for a in L]


# # 将4个子图的概率融合

# In[2]:


def to_sub_result(csv_name):
    #将4个子图的概率融合,此处csv_name没有地址
    result = pd.read_csv('./results_tmp/'+csv_name)
    filename_list = []
    prob_list = []

    for i in range(0, len(result), 4):
        prob_per_image = np.zeros((4, 11))
        prob_final = np.zeros(11)
        # 确定文件名
        filename = result.iloc[i].filename.split('.jpg')[0][:-2]+'.jpg'


        prob = result.iloc[i].probability
        prob = prob.split('[')[1].split(']')[0].split(',') 
        prob_per_image[0] = prob


        prob1 = result.iloc[i+1].probability
        prob1 = prob1.split('[')[1].split(']')[0].split(',') 
        prob_per_image[1] = prob1


        prob2 = result.iloc[i+2].probability
        prob2 = prob2.split('[')[1].split(']')[0].split(',') 
        prob_per_image[2] = prob2


        prob3 = result.iloc[i].probability
        prob3 = prob3.split('[')[1].split(']')[0].split(',') 
        prob_per_image[3] = prob3

        max_value, max_index = get_max(prob_per_image)
        if max_value >= 0.7:
            #print(prob_per_image[max_index[0]])
            prob_final = prob_per_image[max_index[0]]
        elif max_value >= 0.5:
            #prob_per_image = [np.delete(x, [0, max_index[1]])for x in prob_per_image]
            prob_final = np.mean(prob_per_image, axis = 0) 
            prob_final[max_index[1]] = max_value
            prob_final[0] =  1 - np.sum(prob_final[1:]) if 1 - np.sum(prob_final[1:]) > 0 else 0.00001   
        else:
            prob_final = np.mean(prob_per_image, axis = 0)

        filename_list.append(filename + '|' + sub_label_list[0])
        filename_list.append(filename + '|' + sub_label_list[1])
        filename_list.append(filename + '|' + sub_label_list[2])
        filename_list.append(filename + '|' + sub_label_list[3])
        filename_list.append(filename + '|' + sub_label_list[4])
        filename_list.append(filename + '|' + sub_label_list[5])
        filename_list.append(filename + '|' + sub_label_list[6])
        filename_list.append(filename + '|' + sub_label_list[7])
        filename_list.append(filename + '|' + sub_label_list[8])
        filename_list.append(filename + '|' + sub_label_list[9])
        filename_list.append(filename + '|' + sub_label_list[10])

        prob_list.append(round(prob_final[0], 8))
        prob_list.append(round(prob_final[1], 8))
        prob_list.append(round(prob_final[2], 8))
        prob_list.append(round(prob_final[3], 8))
        prob_list.append(round(prob_final[4], 8))
        prob_list.append(round(prob_final[5], 8))
        prob_list.append(round(prob_final[6], 8))
        prob_list.append(round(prob_final[7], 8))
        prob_list.append(round(prob_final[8], 8))
        prob_list.append(round(prob_final[9], 8))
        prob_list.append(round(prob_final[10], 8))

    # 将结果放入sub种
    sub = pd.DataFrame({"filename|defect":filename_list, "probability":prob_list})
    sub.to_csv('./results_tmp/sub_'+csv_name, index = None)


# # ensemble

# In[3]:


def ensemble(csv_list, weights,last_csv):
    #csv_list 为有地址的csv名称列表，last_csv也是有地址的
    for i in range(len(csv_list)):
        if i == 0:
            df1 = pd.read_csv(csv_list[i])
            df1['probability'] = weights[i] * df1.probability
        else:
            df1 = pd.concat([df1, weights[i] * pd.read_csv(csv_list[i]).probability], axis = 1)

    df1['mean'] = df1.probability.sum(axis = 1)
    df_sub = df1[['filename|defect', 'mean']].copy()
    df_sub.columns = ['filename|defect', 'probability']
    df_sub.to_csv(last_csv, index = None)


# In[10]:


def ensemble_pretreatment(csv_list, weights, new_csv_name):
    for i in range(len(csv_list)):
        if i == 0:
            df1 = pd.read_csv(csv_list[i])
            df1['probability'] = weights[i] * df1.probability
        else:
            df1 = pd.concat([df1, weights[i] * pd.read_csv(csv_list[i]).probability], axis = 1)

    df1['mean'] = df1.probability.sum(axis = 1)
    df_sub = df1[['filename', 'mean']].copy()
    df_sub.columns = ['filename', 'probability']
    df_sub.to_csv(new_csv_name, index = None)


# In[15]:


def post_treatment(data_csv, pre_csv, sub_csv):
    #本函数通过read 预处理结果获得最终submit
    data = pd.read_csv(data_csv)
    # 读取预处理csv
    pre_data = pd.read_csv(pre_csv)

    old_prob_list,new_prob_list = [], []
    name_list = []
    flag = np.zeros(10)
    for i in range(0,len(data),11):
        si_prob_list = []
        filename = data['filename|defect'][i].split('|')[0]
        name_list.append(filename)
        old_prob_list.append([data['probability'][i+k] for k in range(11)])
        # 获得filename在预处理csv中的索引
        index = np.where(pre_data['filename']==filename)[0][0]
        a = 1-pre_data['probability'][index]    #a为预处理csv中对应文件的 normal置信度
        b = data['probability'][i]   #b为复赛csv中对应文件的 normal置信度
        new_filename = filename+'_chu'+str(round(a,3))+'_fu'+str(round(b,3))+'.jpg'
        # 判断
        if a > 0.96:
            if b >= 0.7:
                si_prob_list.append(10)
                flag[0]+=1
            elif 0.3 < b <= 0.7:
                si_prob_list.append((a+b)/2)
                flag[1]+=1
            else :
                si_prob_list.append(b)
                flag[2]+=1

    #     elif 0.6<a<=0.8:
    #         if b > 0.6:
    #             si_prob_list.append(max(a,b))
    #         else:
    #             si_prob_list.append(b)
    #             flag[3]+=1

    #     elif 0.3<a<=0.6:
    #         if b <= 0.6:
    #             si_prob_list.append(b/2)
    #             flag[4]+=1
    #         else:
    #             si_prob_list.append(b)
    #             shutil.copy('./test_b/'+filename, './dif/'+new_filename)
    #             flag[5]+=1
    #     elif 0.1<a<=0.3:
    #         if b >= 0.3:
    #             si_prob_list.append(b)
    #             flag[6]+=1
    #         else:
    #             si_prob_list.append(min(a,b))
    #             flag[7]+=1

        elif a<=0.1:
            if b < 0.3:
                si_prob_list.append(0.0001)
                flag[8]+=1
            else:
                si_prob_list.append((b+a)/2)
                flag[9]+=1
        else:
            si_prob_list.append(b)
        # 传入置信度数组中 其他“瑕疵类别置信度值”    
        for j in range(1,11,1):
            si_prob_list.append(data['probability'][i+j])
        # 归一化置信度数组
        new_prob_list.append(normalization(si_prob_list))
        
    filename_list = []
    prob_list = []
    for i in range(len(name_list)):
        filename = name_list[i]
        prob_final = new_prob_list[i]

        filename_list.append(filename + '|' + sub_label_list[0])
        filename_list.append(filename + '|' + sub_label_list[1])
        filename_list.append(filename + '|' + sub_label_list[2])
        filename_list.append(filename + '|' + sub_label_list[3])
        filename_list.append(filename + '|' + sub_label_list[4])
        filename_list.append(filename + '|' + sub_label_list[5])
        filename_list.append(filename + '|' + sub_label_list[6])
        filename_list.append(filename + '|' + sub_label_list[7])
        filename_list.append(filename + '|' + sub_label_list[8])
        filename_list.append(filename + '|' + sub_label_list[9])
        filename_list.append(filename + '|' + sub_label_list[10])
        #print(prob_final)
        prob_list.append(round(prob_final[0], 8))
        prob_list.append(round(prob_final[1], 8))
        prob_list.append(round(prob_final[2], 8))
        prob_list.append(round(prob_final[3], 8))
        prob_list.append(round(prob_final[4], 8))
        prob_list.append(round(prob_final[5], 8))
        prob_list.append(round(prob_final[6], 8))
        prob_list.append(round(prob_final[7], 8))
        prob_list.append(round(prob_final[8], 8))
        prob_list.append(round(prob_final[9], 8))
        prob_list.append(round(prob_final[10], 8))

    # 将结果放入submit中
    sub = pd.DataFrame({"filename|defect":filename_list, "probability":prob_list})
    sub.to_csv(sub_csv, index = None, columns = ["filename|defect","probability"])


# In[6]:


print('ensemble results...')
for i in range(6):
    to_sub_result('densenet201_'+str(i)+'.csv')
    
ensemble(['./results_tmp/sub_densenet201_0.csv',
         './results_tmp/sub_densenet201_1.csv',
         './results_tmp/sub_densenet201_2.csv',
         './results_tmp/sub_densenet201_3.csv',
         './results_tmp/sub_densenet201_4.csv',
         './results_tmp/sub_densenet201_5.csv',],
        [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667],
        './results_tmp/sub_densenet201_last.csv')


# In[11]:


print('ensemble pretreatment results...')
ensemble_pretreatment(['./results_tmp/pretreatment_0.csv',
                       './results_tmp/pretreatment_1.csv',
                       './results_tmp/pretreatment_2.csv',
                       './results_tmp/pretreatment_3.csv',
                       './results_tmp/pretreatment_4.csv',
                       './results_tmp/pretreatment_5.csv',],
                      [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667],
                      './results_tmp/pretreatment_all.csv')


# In[16]:


print('building last submit...')
post_treatment('./results_tmp/sub_densenet201_last.csv','./results_tmp/pretreatment_all.csv','./result.csv')

