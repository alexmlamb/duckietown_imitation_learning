import h5py
import torch
import torch.nn as nn
from utils import to_var
import random
import numpy as np
from data_loader import get_batch
from predictor_models import Predictor

num_images_back = 3
num_targets_forward = 1

print "num images back", num_images_back
print "num targets forward", num_targets_forward

predictor = Predictor(num_images_back, num_targets_forward)

predictor.cuda()

optim = torch.optim.Adam(predictor.parameters(), lr=0.0001)

def norm(inp):
    return inp/255.0 - 0.5

def evaluate(segment,n_batches):

    loss_lst_velocity = []
    loss_lst_heading = []

    for iteration in range(0,n_batches):
        images, targets = get_batch(128, segment, num_images_back, num_targets_forward)

        images = norm(to_var(torch.from_numpy(images)))
        targets = to_var(torch.from_numpy(targets))

        pred = predictor(images)

        loss_v = torch.abs(pred - targets)[:,0::2].mean()
        loss_h = torch.abs(pred - targets)[:,1::2].mean()

        loss_lst_velocity.append(loss_v)
        loss_lst_heading.append(loss_h)

    return (sum(loss_lst_velocity)/len(loss_lst_velocity)).data.cpu().numpy().tolist(), (sum(loss_lst_heading) / len(loss_lst_heading)).data.cpu().numpy().tolist()


if __name__ == "__main__":

    for iteration in range(0,20000):

        images, targets = get_batch(128, "train", num_images_back, num_targets_forward)

        print targets

        raise Exception('done')

        images = norm(to_var(torch.from_numpy(images)))
        targets = to_var(torch.from_numpy(targets))

        pred = predictor(images)

        loss = torch.abs(pred - targets).mean()

        predictor.zero_grad()
        loss.backward()
        optim.step()

        if iteration % 100 == 0:
            print iteration
            print "train velocity/heading", evaluate("train",20)
            print "test velocity/heading", evaluate("test",20)

    print images.shape
    print targets.shape


