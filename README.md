# 雪浪制造AI挑战赛—视觉计算辅助良品检测  复赛名次13 复赛分数0.747
### 预处理
 0.获取测试集索引csv，见 ./data/test_b.csv
 
 1.将所有测试集图片一分4，height为1020，width为1320，并获得索引，见  ./data/test_b_edit.csv
 
 2.将测试集图片分别进行水平和垂直翻转，共形成**3个测试集**。
 
 3.测试集地址为 ./data/test_b_edit  ./data/test_b_edit1 ./data/test_b_edit2。 索引地址为 ./data/test_b.csv  ./data/test_b_edit.csv

### predict
 1.用sh运行densenet201.py文件，此操作将自动在根文件夹中生成 results_tmp，保存过程文件
 
 2.预测将运行6次，调用了同为densenet201网络的 **两个不同训练epochs**的模型，在3个测试集上进行测试。
 
 3.两个模型地址为  **./models/densenet201_0821_256fc_last_epoch.h5**   **./models/densenet201_0824_256fc_last_epoch.h5**

### 辅助判定模型
 1.用sh分别运行初赛网络，inceptionV3和xception，每个网络预测3次，生成辅助判定csv
 
 2.初赛模型地址为 **./models/pretreatment_models/inception_v3_0802_edit.h5**   **../models/pretreatment_models/xception_0731_edit.h5**

### 结果生成
 1.运行result_process，生成 复赛结果csv **./results_tmp/sub_densenet201_last.csv**。  生成 辅助判定csv， **./results_tmp/pretreatment_all.py**
 
 2、通过 辅助判定模型 最终 生成submit， ./result.csv

### 分数说明
 **./results_tmp/sub_densenet201_last.csv**  分数 0.7406
 **./result.csv**  分数 0.7474
