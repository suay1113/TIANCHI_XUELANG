python ./src/data_aug.py
echo "数据增强已完成"


echo "start predict..."
python ./src/densenet201.py

echo "start pretreatment..."
python ./src/pretreatment_inceptionV3.py
python ./src/pretreatment_xception.py

echo "start result_process..."
python ./src/result_process.py

echo "All Done"
