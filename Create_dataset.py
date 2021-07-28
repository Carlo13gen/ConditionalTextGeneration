import math

from sklearn.model_selection import train_test_split

from pycocotools.coco import COCO
import numpy as np

# initialize COCO api for caption annotations
train=[]
annFile_train = './Dataset/captions_train2014.json'
coco_caps_train =COCO(annFile_train)
annIds_train = coco_caps_train.getAnnIds()
anns_train = coco_caps_train.loadAnns(annIds_train)
for x in anns_train:
  cap=x['caption']
  train.append(cap)

# initialize COCO api for caption annotations
val=[]
annFile = './Dataset/captions_val2014.json'
coco_caps=COCO(annFile)
annIds = coco_caps.getAnnIds()
anns = coco_caps.loadAnns(annIds)
for x in anns:
  cap = x['caption']
  val.append(cap)

print('val: '+str(len(val)))
print('train: '+str(len(train)))
lista_tot= val+train
print('somma: '+str(len(lista_tot)))

def cut_strings(s):
    words = s.split(' ')
    split = int(math.ceil(len(words)/2.0))
    return " ".join(words[:split])," ".join(words[split:])


all_x = []
all_y = []
for s in lista_tot:
    first_half, second_half = cut_strings(s)
    all_x.append(first_half)
    all_y.append(second_half)

X = all_x
Y = all_y

train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

x_train = np.array(x_train)
y_train = np.array(y_train)

x_val = np.array(x_val)
y_val = np.array(y_val)

x_test = np.array(x_test)
y_test = np.array(y_test)

np.save('x_train',x_train)
np.save('y_train',y_train)

np.save('x_val',x_val)
np.save('y_val',y_val)

np.save('x_test',x_test)
np.save('y_test',y_test)