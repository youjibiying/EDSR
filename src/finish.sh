#先修改dir_data=../../data/SR  download 会自动下载模型
python main.sh  --data_test Set5+Set14 --data_range 801-900 --scale 4 --pre_train download --test_only --self_ensemble