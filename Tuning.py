import os

#os.system('yes | conda create -n py36 python=3.6')

#os.system('conda activate py36')

os.system('yes | conda install -c conda-forge pycocotools')

os.system('yes | conda install matplotlib')

os.system('yes | conda install -c conda-forge fastBPE')

os.system('yes | conda install -c conda-forge gast==0.2.2')

os.system('yes | conda uninstall tensorflow')

os.system('yes | conda install tensorflow==1.14')

os.system('yes | conda install -c conda-forge gsutil')

os.system('yes | conda install tqdm')

os.system('yes | conda install -c conda-forge nltk')

os.system('python Create_dataset.py')

os.system('git clone https://github.com/salesforce/ctrl.git')

os.system('mv ./Training_mod.py ./ctrl/training_utils/')

os.chdir('./ctrl')

#os.system(' (echo "import os" ; echo "print(os.listdir('/anaconda/envs/py36/lib/python3.6/site-packages/tensorflow_estimator/python/estimator/'))") | python ')

os.system('patch -b /anaconda/envs/py36/lib/python3.6/site-packages/tensorflow_estimator/python/estimator/keras.py estimator.patch')

os.chdir('./training_utils')

os.system('gsutil -m cp -r gs://sf-ctrl/seqlen256_v1.ckpt/ .')

for i in range(4):
    os.system('mv ../../train'+str(i)+'.txt .')
    print('Make TFRecord of training file: '+str(i)+'\n')
    os.system('python ./make_tf_records.py --text_file train'+str(i)+'.txt --control_code caption --sequence_len 256')
    print('Finish TFRecord of training file: ' + str(i) + '\n')

lr = [1e-2, 1e-1, 10]
iterations = [1, 5, 10]



os.system('mv ../../seed_file.txt ../')

os.system('mv ../../Generate_captions.py ../')

training_params = [(i, j) for i in lr for j in iterations]
for pair in training_params:
    os.system("python ./Training_mod.py --model_dir seqlen256_v1.ckpt --iterations " + str(pair[1]) + " --learning_rate " + str(pair[0]))

    os.chdir('..')

    temperatures = [0.1, 0.2, 0.5]
    topk = [1, 5, 10]
    nucleus = [0.1, 0.5, 0.9]
    generation_topk_params =  [(i, j) for i in temperatures for j in topk ]
    for topk_pair in generation_topk_params:
        os.system("python ./Generate_captions.py --model_dir ./training_utils/seqlen256_v1.ckpt --temperature " + str(topk_pair[0]) + " --topk " + str(topk_pair[1]) + " --print_once  --input_file ./seed_file.txt")

        os.chdir('..')
        os.system('python ./Evaluation.py ./reference.txt ./ctrl/output.txt ./score.txt %f %d %f %d %s' % (pair[0], pair[1], topk_pair[0], topk_pair[1], "NAN"))
        os.chdir("./ctrl")

    generation_nucleus_params =  [(i, j) for i in temperatures for j in nucleus ]
    for nucleus_pair in generation_nucleus_params:
        os.system("python ./Generate_captions.py --model_dir ./training_utils/seqlen256_v1.ckpt --temperature " + str(nucleus_pair[0]) +  " --print_once --nucleus " + str(nucleus_pair[1]) + " --input_file ./seed_file.txt")

        os.chdir('..')
        os.system('python ./Evaluation.py ./reference.txt ./ctrl/output.txt ./score.txt %f %d %f %s %f' % (pair[0], pair[1], nucleus_pair[0], "NAN", nucleus_pair[1]))
        os.chdir("./ctrl")

    os.chdir('./training_utils')
