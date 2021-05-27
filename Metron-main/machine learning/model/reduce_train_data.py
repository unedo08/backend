import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm


x_train=np.load('dataset/x_train_np.npy')
x_cv=np.load('dataset/x_cv_np.npy')
y_train=np.load('dataset/y_train_np.npy')
y_cv=np.load('dataset/y_cv_np.npy')

train_sizes=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
cv_sizes=[0.4, 0.5, 0.6, 0.7]

print('generating reduced train sets')
train_reduced=[]
train_reduced_labels=[]
for train_size in tqdm(train_sizes):
    x_train_reduced, x_cv_reduced, y_train_reduced, y_cv_reduced = train_test_split(x_train, y_train, test_size=train_size,stratify=y_train)
    train_reduced.append(x_train_reduced)
    train_reduced_labels.append(y_train_reduced)
print('done')

print('generating reduced cv sets')
cv_reduced=[]
cv_reduced_labels=[]
for cv_size in tqdm(cv_sizes):
    x_cv_reduced, x_test_reduced, y_cv_reduced, y_test_reduced = train_test_split(x_cv, y_cv, test_size=cv_size, stratify=y_cv)
    cv_reduced.append(x_cv_reduced)
    cv_reduced_labels.append(y_cv_reduced)
print('done')

os.mkdir('train_data_reduced')

print('saving')
for size, reduced in zip(train_sizes, train_reduced):
    np.save('train_data_reduced/x_train_' + str(size) + '.npy', reduced)

for size, reduced in zip(train_sizes, train_reduced_labels):
    np.save('train_data_reduced/y_train_' + str(size) + '.npy', reduced)

for size, reduced in zip(cv_sizes, cv_reduced):
    np.save('train_data_reduced/x_cv_' + str(size) + '.npy', reduced)

for size, reduced in zip(cv_sizes, cv_reduced_labels):
    np.save('train_data_reduced/y_cv_' + str(size) + '.npy', reduced)
print('done')