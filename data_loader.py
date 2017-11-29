import h5py
import random
import numpy as np

fh = h5py.File('/data/lisa/data/duckie_data/data.hdf5','r')

all_images = fh['image']
all_targets = fh['control']



percent_train = 0.75

train_cutoff = int(all_images.shape[0]*percent_train)

num_points = all_images.shape[0]

def get_example_single(segment, n_images_back, n_targets_forward):
    
    if segment == "train":
        
        startstep = random.randint(n_images_back, train_cutoff - n_targets_forward)

    elif segment == "test":

        startstep = random.randint(train_cutoff + n_images_back, num_points - n_targets_forward)

    else:
        raise Exception()

    #image = all_images[startstep-n_images_back+1:startstep+1]

    image_lst = []
    target_lst = []

    for j in range(n_images_back):
        image = all_images[startstep-j]
        image = image.reshape(1, 60,80,3)
        image_lst.append(image)

    for j in range(n_targets_forward):
        target = all_targets[startstep+j].reshape(1,2)
        target_lst.append(target)


    images = np.concatenate(image_lst,3).transpose(0,3,1,2).astype('float32')
    targets = np.concatenate(target_lst,1).astype('float32')

    return images, targets

def get_batch(batch_size, segment, n_images_back, n_targets_forward):

    assert segment in ["train", "test"]

    image_lst = []
    target_lst = []

    for i in range(batch_size):
        images, targets = get_example_single(segment, n_images_back, n_targets_forward)
        image_lst.append(images)
        target_lst.append(targets)

    images = np.concatenate(image_lst, 0)
    targets = np.concatenate(target_lst, 0)

    return images,targets


