#Script to download prfile images of Kickstarter campaigns. Code reads links in kickstarter dataset downoladed from https://webrobots.io/kickstarter-datasets/

import csv
import sys
import os
import urllib
from datetime import datetime
import shutil
import os
import glob
import os.path as osp
import cv2

positive_train_samples_folder='c:/work/crowdfunding/codeImageClassification/'       #'e:/work/clanci/2021/crowdfunding/images/train/positive/'
negative_train_samples_folder='c:/work/crowdfunding/codeImageClassification/'   #   'e:/work/clanci/2021/crowdfunding/images/train/negative/'

positive_test_samples_folder='e:/work/clanci/2021/crowdfunding/images/test/positive/'
negative_test_samples_folder='e:/work/clanci/2021/crowdfunding/images/test/negative/'

counter=0
counter_positive=0
counter_negative=0

def move_images(source_dir,target_dir):

    file_names = os.listdir(source_dir)
    N_images =int(0.8 * len(file_names))
    for file_name in file_names[-N_images:]:
        shutil.move(os.path.join(source_dir, file_name), target_dir)


def move_corrupted_images(source_dir,target_dir):
    # utility fo filter corrupted images
    file_names = os.listdir(source_dir)
    file_names=glob.glob(osp.join(source_dir, '*.png'))
    for file_name in file_names:
        img=cv2.imread(file_name)
        if img is None:
            shutil.move(os.path.join(source_dir, file_name), target_dir)

def readFile(fileName):
    f = open(fileName, "r",encoding="utf8")
    reader = csv.reader(f)
    counter_row=0
    global counter,counter_positive,counter_negative
    for row in reader:
        if counter_row==0:
            state_index=row.index('state')
            photo_index=row.index('photo')
            launched_index=row.index('launched_at')
            counter_row = counter_row + 1
            continue

        l = row[photo_index].split('"')
        link=l[7]
        if row[state_index]=='successful':
            try:
                urllib.request.urlretrieve(link, positive_train_samples_folder+str(counter_positive)+'.png')
            except Exception as e:
                print(e)
                continue  # conti
            counter_positive=counter_positive+1
        if row[state_index] == 'failed':
            try:
                 urllib.request.urlretrieve(link, negative_train_samples_folder + str(counter_negative)+'.png')
            except Exception as e:
                print(e)
                continue  # cont
            counter_negative = counter_negative + 1
        counter=counter+1
        if counter_row%10==0:
            print(counter_row)
        counter_row=counter_row+1
    f.close()



def main():
    #folder with csv fiels downloaded from https://webrobots.io/kickstarter-datasets/
    folder='e:\\work\\clanci\\2021\\crowdfunding\\downloads\\2_21\\'    #18_5_18
    listing = os.listdir(folder)

    # download images
    for file in listing:
        print(counter)
        readFile(folder+file)
if __name__ == '__main__':
    main()




