import os
from os import listdir
from os.path import isfile, join
import cPickle as pickle
import json

"""
File would be to pre-process Train, Test (Dataset of MS-COCO2014) 

The file is yet to be completed
"""

def preprocess_images(img_folder):
    onlyfiles = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]
    train_imgs = {}
    for i in onlyfiles:
        img = np.array(image.load_img(img_folder + '/' + i, target_size=(224,224)))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img = preprocess_input(x)
        train_imgs[f] = x
	train_imgs = "Train_imgs_file.p"
	with open(train_imgs, 'r') as f:
	    pickle.dump(train_imgs, f)

def prep_train_QA(mquestion, annotationfile):
    questions = json.load(open(mquestion))
    annotations = json.load(open(annotationfile))

if __name__ == "__main__":
#    preprocess_images('Train')
#    prep_train_QA("MultipleChoice_mscoco_train2014_questions.json", "mscoco_train2014_annotations.json")
    data = json.load(open("mscoco_train2014_annotations.json"))
    print(data['annotations'][0])
