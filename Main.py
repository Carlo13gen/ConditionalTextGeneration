import os

#from ipython_genutils.py3compat import execfile
os.system('pip install pycocotools')

os.system('python Create_dataset.py')

os.system('git clone https://github.com/salesforce/ctrl.git')

os.chdir('./ctrl/training_utils')

os.system('pip install gsutil')

os.system('gsutil -m cp -r gs://sf-ctrl/seqlen256_v1.ckpt/ .')

os.chdir('../..')

os.system('yes | conda create -n py27 python=2.7')

os.system('conda activate py27')

os.system('yes | conda install matplotlib')

os.system('yes | conda install -c conda-forge fastBPE')

os.system('yes | conda install -c conda-forge gast==0.2.2')

os.system('yes | conda uninstall tensorflow')

os.system('yes | conda install tensorflow==1.14')

os.system(' (echo "import os" ; echo "print(os.listdir(\'/anaconda/envs/py27/lib/python2.7/site-packages/tensorflow_estimator/python/estimator/\'))") | python ')

os.chdir('./ctrl')

os.system('patch -b /anaconda/envs/py27/lib/python2.7/site-packages/tensorflow_estimator/python/estimator/keras.py estimator.patch')

os.system('yes | conda install tqdm')

os.chdir('./ctrl/training_utils')

for i in range(4):
    os.system('mv ../../train'+str(i)+'.txt .')
    print('Make TFRecord of training file: '+str(i)+'\n')
    os.system('python ./make_tf_records.py --text_file train'+str(i)+'.txt --control_code caption --sequence_len 256')
    print('Finish TFRecord of training file: ' + str(i) + '\n')

os.system('python ./training.py --model_dir seqlen256_v1.ckpt --iterations 5')

os.chdir('..')

os.system('python ./generation.py --model_dir ./training_utils/seqlen256_v1.ckpt --temperature 0.2 --topk 5')



