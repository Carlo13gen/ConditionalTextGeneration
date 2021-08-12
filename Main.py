import os

#from ipython_genutils.py3compat import execfile

os.system('unzip ./Dataset/coco_annotation.zip -d ./Dataset')
os.system('unzip ./annotations-test/image_info_test2014.zip -d ./annotations-test')

os.system('conda create -n py27 python=2.7')

os.system('conda activate py27')

os.system('conda install -c conda-forge pycocotools')

os.system('conda install matplotlib')

os.system('python Create_dataset.py')

os.system('git clone https://github.com/salesforce/ctrl.git')

os.system('conda install -c conda-forge fastBPE')

os.system('conda install -c conda-forge gast==0.2.2')

os.system('conda uninstall tensorflow')

os.system('conda install tensorflow==1.14')

os.system(' (echo "import os" ; echo "print(os.listdir(\'/anaconda/envs/py36/lib/python3.6/site-packages/tensorflow_estimator/python/estimator/\'))") | python ')

os.chdir('./ctrl')

os.system('patch -b /anaconda/envs/py36/lib/python3.6/site-packages/tensorflow_estimator/python/estimator/keras.py estimator.patch')

os.system('conda install -c conda-forge gsutil')

os.system('conda install tqdm')

os.system('gsutil -m cp -r gs://sf-ctrl/seqlen256_v1.ckpt/ .')

os.chdir('./ctrl/training_utils')

os.system('python ./make_tf_records.py --text_file ../../train.txt --control_code caption --sequence_len 256')

os.system('python ./training.py --model_dir ../seqlen256_v1.ckpt --iterations 500')

os.chdir('..')

os.system('python ./generation.py --model_dir ./training_utils/seqlen256_v1.ckpt --temperature 0.2 --topk 5')



