from random import shuffle
from glob import glob
import natsort
import json
import matplotlib.pyplot as plt
from dtw import dtw
import os
import argparse
import pandas as pd
import csv
import numpy as np


ap = argparse.ArgumentParser()

ap.add_argument("-csv_files_path","--csv_files_path",required=True,
    help="path of input iamge directories")

args = vars(ap.parse_args())
data = pd.read_csv(args['csv_files_path'], error_bad_lines=False)

skeleton_frames=list(data['skeleton images path'])
inp_sentences = list(data['input sentence'])
words = list(data['word'])

#path_skeleton_frame
#inp_sentence
#individualword

#to get all the body joints using openpose


def getjoints(jsonfile):
  with open(jsonfile) as datafile:
    data = json.load(datafile)
  data2=np.array(data['part_candidates'][0]['2']) #right shoulder
  data3=np.array(data['part_candidates'][0]['3']) #right elbow
  data4=np.array(data['part_candidates'][0]['4']) #right Wrist
  data5=np.array(data['part_candidates'][0]['5']) #left shoulder
  data6=np.array(data['part_candidates'][0]['6']) #left elbow
  data7=np.array(data['part_candidates'][0]['7']) #left Wrist
  x = [data2[0],data3[0],data4[0],data5[0],data6[0],data7[0]]
  y = [data2[1],data3[1],data4[1],data5[1],data6[1],data7[1]]
  return((x,y))


def write(row):
  with open('skeleton.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow(row)
  csvFile.close()
row  = ['word','skeleton path for word']
write(row)



def getskeletonframes(word,inp_sent,path):
    #print(path)
    path = path+'/'
    tx = []
    ty = []

    for i in natsort.natsorted(glob(path+ '*.json')):
      x,y = getjoints(i)
      tx.append(x)
      ty.append(y)

    image_paths = [img for img in natsort.natsorted(glob(path+ '*.png'))]
    d = []
    for i in range(1,len(tx)):
      d.append(np.sqrt(((np.array(tx[i-1])-np.array(tx[i]))**2)+((np.array(ty[i-1])-np.array(ty[i]))**2)))

    d0 = [d[i][0] for i in range(len(d))]
    d1 = [d[i][1] for i in range(len(d))]
    d2 = [d[i][2] for i in range(len(d))]
    d3 = [d[i][3] for i in range(len(d))]
    d4 = [d[i][4] for i in range(len(d))]
    d5 = [d[i][5] for i in range(len(d))]

    body = {1:'right shoulder',
            2:'right elbow',
            3:'right wrist',
            4:'left shoulder',
            5:'left elbow',
            6:'left wrist'}

    dd= [d0,d1,d2,d3,d4,d5]



    #dynamic time wrapping


    y = np.arange(len(inp_sent.split(' '))) #[0,1,2]
    x1 = np.array(d1).reshape(-1, 1)
    #print(np.array(x1))
    arr =x1
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]
    x1 = arr
    euclidean_norm = lambda x1, y: np.abs(x1 - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(x1, y, dist=euclidean_norm)


    x2 = np.array(d4).reshape(-1, 1)
    arr = x2
    rightwrist = path[1] 
    arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]
    x2 = arr
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)

    euclidean_norm = lambda x2, y: np.abs(x2 - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(x2, y, dist=euclidean_norm)
    leftwrist = path[1]

    val = min(len(rightwrist),len(leftwrist))
    final = np.maximum(np.array(rightwrist[0:val]),np.array(leftwrist[0:val]))
    f = {v:sen for v, sen in enumerate(inp_sent.split(' '))}
    frames_data = [f[i] for i in final]
    #word = 'JOHN'
    frames_data_pos = [j for j,i in enumerate(frames_data) if i == word] #prints [0,1,2]
    img_data = [image_paths[i] for i in frames_data_pos]
    #print(img_data)

    print('for word ',word,' no of frames is ',len(img_data))
    
    for i in img_data:
        rows = [word,i]
        write(rows)
        #print(row)      
skeleton_frames=list(data['skeleton images path'])
inp_sentences = list(data['input sentence'])
words = list(data['word'])

for i,j,k in zip(words,inp_sentences,skeleton_frames):
    getskeletonframes(i,j,k)                   

