# 雪浪制造AI挑战赛—视觉计算辅助良品检测  复赛名次13 复赛分数0.747
## 项目说明 
- 参赛地址：[雪浪制造AI挑战赛](https://tianchi.aliyun.com/competition/introduction.htm?spm=a2c22.11695015.1131732.1.4ea25275NNvZuf&raceId=231666) 
- 本项目中仅包括了预测代码，在复赛测试集上将到达0.74左右的分数，并由于采用了基于规则的结果融合，最终成绩会有所波动。
- 由于数据集划分问题，本次比赛训练出的模型十分诡异，因此没有复现网络训练过程，网络结构在测试代码里有全部体现。
- 为清晰展现思路，上传整理的PPT材料，推荐使用Office 2013以上版本打开

## 思想说明
- 训练集与测试集采用了4块子图的裁剪方法，即height为1020，width为1320，随后resize到512的尺度进行输入。

  比赛测试了包括 **densenet201**  **inception_resnet_V2**等在内的多个大型网络，最终选择了两个表现较好的densenet201进行融合。
- 比赛对于有瑕疵的训练集子图进行了增强，增强主要包括垂直/水平翻转、裁剪范围变化。
- 比赛采用了测试增强（Test set augmentation），对于测试集预测三次（1+2），获得了0.0123的分数提升
- 比赛引入了初赛二分类的辅助信息，将初赛的normal置信度与复赛12分类结果进行了基于规则的融合，获得了0.0290的提升

## 项目结构
### 脚本运行方式
- 将官网的xuelang_round2_test_b_201808031.zip.zip 文件，解压后修改名称为‘test_b’，放在./data目录下
- 将下载好的训练模型放在./models文件夹中，
- 运行run.sh文件，将生成./result.csv文件，为最终结果。如果运行失败，请参照以下具体步骤，寻找错误原因

### 预处理
- 获取测试集索引csv，见 ./data/test_b.csv
- 将所有测试集图片机型裁剪，裁剪后height为1020，width为1320，保证每块子图有重叠部分，并获得索引，见  ./data/test_b_edit.csv
- 将测试集图片分别进行水平和垂直翻转，共形成**3个测试集**。
- 测试集地址为 ./data/test_b_edit  ./data/test_b_edit1 ./data/test_b_edit2。 索引地址为 ./data/test_b.csv  ./data/test_b_edit.csv

### predict
- 请在百度云下载训练过的权重，地址:[百度云](https://pan.baidu.com/s/1QG8xXBdr3wbguiy_eeuVdg)
- 用sh运行densenet201.py文件，此操作将自动在根文件夹中生成 results_tmp，保存过程文件
- 预测将运行6次，调用了同为densenet201网络的 **两个不同训练epochs**的模型，在3个测试集上进行测试。
- 两个模型地址为  **./models/densenet201_0821_256fc_last_epoch.h5**   **./models/densenet201_0824_256fc_last_epoch.h5**

### 辅助判定模型
- 用sh分别运行初赛网络，inceptionV3和xception，每个网络预测3次，生成辅助判定csv
- 初赛模型地址为 **./models/pretreatment_models/inception_v3_0802_edit.h5**   **../models/pretreatment_models/xception_0731_edit.h5**

### 结果生成
- 运行result_process，生成 复赛结果csv **./results_tmp/sub_densenet201_last.csv**。  生成 辅助判定csv， **./results_tmp/pretreatment_all.py**
- 通过 辅助判定模型 最终 生成submit， ./result.csv

### 分数说明
 **./results_tmp/sub_densenet201_last.csv**  分数 0.7406
 
 **./result.csv**  分数 0.7474
