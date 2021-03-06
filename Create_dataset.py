import math
#import numpy as np
#import pandas as pd
from pycocotools.coco import COCO
import json
import os

def cut_strings(s):
    words = s.split(' ')
    split = int(math.ceil(len(words)/2.0))
    return " ".join(words[:split])," ".join(words[split:])

os.system('unzip ./Dataset/coco_annotation.zip -d ./Dataset')
os.system('unzip ./annotations-test/image_info_test2014.zip -d ./annotations-test')

# initialize COCO api for caption annotations
train= {}
annFile_train = './Dataset/captions_train2014.json'
coco_caps_train =COCO(annFile_train)
annIds_train = coco_caps_train.getAnnIds()
anns_train = coco_caps_train.loadAnns(annIds_train)
i = 0
for x in anns_train:
    cap=x['caption']
    im_id = x['image_id']
    train[x['id']] = {"cap" : cap, "im_id" : im_id}


#saving all categories and supercategories [category:supercategory]
test_im_cat = "./annotations-test/image_info_test2014.json"
file = open(test_im_cat,"r")
j_data = json.load(file)

categories = {}
for item in j_data['categories']:
 categories[item['name']] = item['supercategory']
print(categories)


# initialize COCO api for caption annotations
val={}
annFile = './Dataset/captions_val2014.json'
coco_caps=COCO(annFile)
annIds = coco_caps.getAnnIds()
anns = coco_caps.loadAnns(annIds)
for x in anns:
    cap = x['caption']
    im_id = x['image_id']
    val[x['id']] = {"cap": cap, "im_id": im_id}

print('val: '+str(len(val)))
print('train: '+str(len(train)))
training_data = {**train, **val}
print('somma: '+str(len(training_data)))

#create 4 train dataset 10000 record
for j in range(4):
    train_id = list(training_data.keys())[j*10000:(j+1)*10000]
    train_file = open("train"+str(j)+".txt", "w")
    for idx in train_id:
        train_file.write(training_data[idx]['cap'].strip('\n') +'\n')
    train_file.close()


#create seed and reference file(complete caption for evaluation)
seed_id = list(training_data.keys())[40000:40200]
seed_file = open("seed_file.txt", "w")
reference_file = open('reference.txt','w')
for id in seed_id:
    first_half, second_half = cut_strings(training_data[id]['cap'])
    reference_file.write(training_data[id]['cap']+'\n')
    seed_file.write("caption " + first_half+'\n')
seed_file.close()
reference_file.close()
