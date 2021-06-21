import cv2
import os
import numpy as np
from skimage.util import random_noise
cnt = 0 
file_name = os.listdir('images')

for i in file_name:
    cnt = cnt + 1
    org_file = 'images/' + str(i)
    img_data = cv2.imread(org_file, 0)
    dim = (100, 100)
    blurred = random_noise(img_data, mode='s&p', amount=0.025)
    resized = cv2.resize(blurred, dim, interpolation = cv2.INTER_AREA)
    resized = (resized*255).astype('uint8')
    
    file_name = 'gaussian_2/' + str(cnt) +'.jpg'
    cv2.imwrite(file_name, resized) 
    if cnt == 4000:
        break

folder_list = os.listdir('NImages')
cnt = 0
for j in folders:
    folder_name = 'NImages/' + str(j)
    file_name = os.listdir(folder_name)
    for i in file_name:
        cnt = cnt + 1
        org_file = folder_name +'/' + str(i)
        img_data = cv2.imread(org_file, 0)
        
        dim = (100, 100)
        try:
            blurred = random_noise(img_data, mode='s&p', amount=0.025)
            resized = cv2.resize(blurred, dim, interpolation = cv2.INTER_AREA)
            resized = (resized*255).astype('uint8')
            file_name = 'non_facegaussian2/' + str(cnt) +'.jpg'
            cv2.imwrite(file_name, resized) 
            
        except:
            continue

file_list = os.listdir('gaussian_2/')
cnt = 0
train_tuple = ()
for i in file_list:
    file_name = 'gaussian_2/' + str(i)
    img_data = cv2.imread(file_name,0)
    tr_list = list(train_tuple)
    tr_list.append((np.array(img_data),1))
    train_tuple = tuple(tr_list)
    cnt = cnt+1

face_list = list(train_tuple)
train_face, test_face = tuple(face_list[:3000]), tuple(face_list[3000:])

    
file_list = os.listdir('non_facegaussian2/')
cnt = 0
train_tuple = ()
for i in file_list:
    file_name = 'non_facegaussian2/' + str(i)
    img_data = cv2.imread(file_name,0)
    tr_list = list(train_tuple)
    tr_list.append((np.array(img_data),0))
    train_tuple = tuple(tr_list)
    cnt = cnt+1

nonface_list = list(train_tuple)
nontrain_face, nontest_face = nonface_list[:3000], nonface_list[3000:]

train_data = list(train_face)
final_train = np.concatenate((train_data, nontrain_face), axis = 0)
print(np.shape(final_train))

test_data = list(test_face)
final_test = np.concatenate((test_data, nontest_face), axis = 0)
print(np.shape(final_test))

import pickle
with open("training_gauss_new.pkl", "wb") as file:
    pickle.dump(final_train, file)

import pickle
with open("test_gauss_new.pkl", "wb") as file:
    pickle.dump(final_test, file)


    