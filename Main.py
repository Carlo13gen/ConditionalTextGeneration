import os

from ipython_genutils.py3compat import execfile

os.system('unzip ./Dataset/coco_annotation.zip -d ./Dataset')
os.system('unzip ./annotations-test/image_info_test2014.zip -d ./annotations-test')

os.system('pip install mask')

os.system('python Create_dataset.py')

os.system('git clone https://github.com/salesforce/ctrl.git')

os.system('pip install fastBPE')

os.system('cd ./ctrl/training_utils')

os.system('python ./make_tf_records.py --text_file ../../train.txt --control_code Caption --sequence_len 256')

os.system('gsutil -m cp -r gs://sf-ctrl/seqlen256_v1.ckpt/ .')

os.system('python ./training.py --model_dir ./seqlen256_v1.ckpt --iterations 500')

os.system('cd ..')

os.system(' python ./generation.py --model_dir ./training_utils/seqlen256_v1.ckpt --temperature 0.4 --topk 4')



