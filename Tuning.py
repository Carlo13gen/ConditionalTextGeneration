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
counter = 0
for pair in training_params:
    os.system("python ./Training_mod.py --model_dir seqlen256_v1.ckpt --iterations " + str(pair[1]) + " --learning_rate " + str(pair[0]))

    os.chdir('..')

    temperatures = [0.1, 0.2, 0.5]
    topk = [1, 5, 10]
    penalty = [1, 1.2, 2]
    generation_params =  [(i, j, k) for i in temperatures for j in topk for k in penalty ]
    for triplet in generation_params:
        os.system("python ./Generate_captions.py --model_dir ./training_utils/seqlen256_v1.ckpt --temperature " + str(triplet[0]) + " --topk " + str(triplet[1]) + " --penalty " + str(triplet[2]) + " --print_once --input_file ./seed_file.txt")

        os.chdir('..')
        os.system('python ./Evaluation.py ./reference.txt ./ctrl/output.txt ./score.txt %f %d %f %d %f' % (pair[0], pair[1], triplet[0], triplet[1], triplet[2]))
        os.chdir("./ctrl")



    os.chdir('./training_utils')
    print("End of step %d/%d of training"%(counter+1, 9))

print("END OF TUNING")