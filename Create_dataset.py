import math

import pandas as pd
from sklearn.model_selection import train_test_split

from pycocotools.coco import COCO
import json
#import numpy as np

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



test_im_cat = "./annotations-test/image_info_test2014.json"
file = open(test_im_cat,"r")
j_data = json.load(file)
#print(j_data['images']['id'][2])

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
lista_tot= {**train, **val}
print('somma: '+str(len(lista_tot)))

def cut_strings(s):
    words = s.split(' ')
    split = int(math.ceil(len(words)/2.0))
    return " ".join(words[:split])," ".join(words[split:])


train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

df = pd.DataFrame(lista_tot)
training_data = df.sample(frac=train_ratio, random_state=25)
testing_data = df.drop(training_data.index)

print('train: '+str(len(training_data)))
print('test: '+str(len(testing_data)))

test_data = testing_data.values.tolist()
train_data = training_data.values.tolist()

all_x = []
all_y = []
for s in test_data:
    first_half, second_half = cut_strings(s[0])  #poichè test data è una lista di liste bisogna fare accesso alla lista interna
    all_x.append(first_half)
    all_y.append(second_half)

X = all_x
Y = all_y

x_val, x_test, y_val, y_test = train_test_split(X,Y, test_size=test_ratio/(test_ratio + validation_ratio))

a_file = open("train.txt", "w")
for row in train_data:
    a_file.write(row[0]+'\n')
a_file.close()

a_file = open("x_val.txt", "w")
for row in x_val:
    a_file.write(row+'\n')
a_file.close()

a_file = open("y_val.txt", "w")
for row in y_val:
    a_file.write(row + '\n')
a_file.close()

a_file = open("x_test.txt", "w")
for row in x_test:
    a_file.write(row+'\n')
a_file.close()

a_file = open("y_test.txt", "w")
for row in y_test:
    a_file.write(row+'\n')
a_file.close()