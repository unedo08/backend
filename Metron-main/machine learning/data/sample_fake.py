from multiprocessing import Process, Manager
from imageio import imread
import pickle
import numpy as np
import os
from tqdm import tqdm

fake_path = 'training/fake/'
pristine_path = 'training/pristine/'
mask_path = fake_path + 'masks/'

def count_255(mask):
    i=0
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row,col]==255:
                i+=1
    return i

def sample_fake(img, mask):
    kernel_size = 64
    stride = 32

    samples = []

    for y_start in range(0, img.shape[0] - kernel_size + 1, stride):
        for x_start in range(0, img.shape[1] - kernel_size + 1, stride):

            c_255 = count_255(mask[y_start:y_start + kernel_size, x_start:x_start + kernel_size])

            if (c_255 > 1600) and (kernel_size * kernel_size - c_255 > 1600):
                samples.append(img[y_start:y_start + kernel_size, x_start:x_start + kernel_size, :3])

    return samples


def process(batch, common_list, images, masks):

    for img, mask in zip(images, masks):
        samples=sample_fake(img, mask)
        for s in samples:
            common_list.append(s)
        print('Number of samples = ' + str(len(samples)))

    print(f'batch {batch} completed')


def main():

    x_train_masks = []
    for i in range(9):
        with open('pickle/images/x_train_masks_' + str(i) + '.pickle', 'rb') as f:
            x_train_masks.extend(pickle.load(f))

    with open('pickle/images_names/x_train_fakes_names.pickle', 'rb') as f:
        x_train_fakes_names = pickle.load(f)

    x_train_fake_images = []
    for img in x_train_fakes_names:
        x_train_fake_images.append(imread(fake_path + img))

    m = Manager()
    common_list = m.list()

    processes = []
    for batch in range(9):
        processes.append(Process(target=process, args=(batch, common_list, x_train_fake_images[batch*40:(batch+1)*40],
                                                       x_train_masks[batch*40:(batch+1)*40])))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    samples_fake_np = common_list[0][np.newaxis, :, :, :]
    for fake_sample in common_list[1:]:
        samples_fake_np = np.concatenate((samples_fake_np, fake_sample[np.newaxis, :, :, :3]), axis=0)

    print('done')
    np.save('sample_images/k64 grayscale 40percent stride32/sample_fakes_np.npy', samples_fake_np)


if __name__ == '__main__':
    main()
