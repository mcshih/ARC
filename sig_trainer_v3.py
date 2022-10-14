import os
import sys
import argparse
import torch
from torch.autograd import Variable
from datetime import datetime, timedelta

import batcher
from sig_dataloader import SigDataset
#from batcher import Batcher
import models
from models import ArcBinaryClassifier
from torchvision.utils import save_image, make_grid
from module.transformation_simple import SpatialTransformerNetwork
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to ARC')
parser.add_argument('--glimpseSize', type=int, default=8, help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=128, help='number of hidden states in ARC controller')
parser.add_argument('--numGlimpses', type=int, default=6, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for saving'
                                                 ' model checkpoints in a separate folder.')
parser.add_argument('--load', default=None, help='the model to load from. Start fresh if not specified.')
parser.add_argument('--data', type=str, default="./../ChiSig", help='data path')
parser.add_argument('--stn', action='store_true', help='stn net')
parser.add_argument('--test_only', action='store_true', help='stn net')

def get_pct_accuracy(pred: Variable, target) -> int:
    hard_pred = (pred > 0.5).int()
    #correct = (hard_pred == target).sum().data[0]
    correct = (hard_pred == target).sum().data
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy

def display(opt, image1, mask1, image2, mask2, name="hola.png"):
    _, ax = plt.subplots(1, 2)
    
    # a heuristic for deciding cutoff
    masking_cutoff = 2.4 / (opt.glimpseSize)**2

    mask1 = (mask1 > masking_cutoff).data.cpu().numpy()
    mask1 = np.ma.masked_where(mask1 == 0, mask1)

    mask2 = (mask2 > masking_cutoff).data.cpu().numpy()
    mask2 = np.ma.masked_where(mask2 == 0, mask2)

    ax[0].imshow(image1.data.cpu().numpy(), cmap=mpl.cm.bone)
    ax[0].imshow(mask1, interpolation="nearest", cmap=mpl.cm.jet_r, alpha=0.7)

    ax[1].imshow(image2.data.cpu().numpy(), cmap=mpl.cm.bone)
    ax[1].imshow(mask2, interpolation="nearest", cmap=mpl.cm.ocean, alpha=0.7)

    images_path = os.path.join("visualization", opt.name)
    os.makedirs(images_path, exist_ok=True)
    
    plt.savefig(os.path.join(images_path, name))

def train(opt):

    if opt.cuda:
        batcher.use_cuda = True
        models.use_cuda = True

    if opt.name is None:
        # if no name is given, we generate a name from the parameters.
        # only those parameters are taken, which if changed break torch.load compatibility.
        opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                        "cuda" if opt.cuda else "cpu")

    print("Will start training {} with parameters:\n{}\n\n".format(opt.name, opt))

    # make directory for storing models.
    models_path = os.path.join("saved_models", opt.name)
    os.makedirs(models_path, exist_ok=True)
    with open(os.path.join(models_path, 'args.txt'),'w') as f:
        f.write(' '.join(str(x) for x in sys.argv))
        json.dump(opt.__dict__,f,indent=4)

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)
    if opt.stn:
        STN_net = SpatialTransformerNetwork(I_size=(opt.imageSize, opt.imageSize),
                                            I_r_size=(opt.imageSize, opt.imageSize),
                                            I_channel_num=1)

    if opt.cuda:
        discriminator.cuda()
        if opt.stn:
            STN_net.cuda()

    # load from a previous checkpoint, if specified.
    if opt.load is not None:
        discriminator.load_state_dict(torch.load(os.path.join(models_path, opt.load)))
        if opt.stn:
            STN_net.load_state_dict(torch.load(os.path.join(models_path, opt.load+"_STN")))

    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.cuda:
        bce = bce.cuda()

    optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=opt.lr)
    if opt.stn:
        optimizer_stn = torch.optim.Adam(params=STN_net.parameters(), lr=opt.lr)

    # load the dataset in memory.
    sigdataset_train = SigDataset(opt.data, train=True, image_size=opt.imageSize)
    sigdataset_test = SigDataset(opt.data, train=False, image_size=opt.imageSize)
    
    train_loader = DataLoader(sigdataset_train, batch_size=opt.batchSize, shuffle=True)
    test_loader = DataLoader(sigdataset_test, batch_size=1, shuffle=False)
    #loader = Batcher(batch_size=opt.batchSize, image_size=opt.imageSize)

    # ready to train ...
    best_validation_loss = None
    saving_threshold = 1.02
    last_saved = datetime.utcnow()
    save_every = timedelta(minutes=59)

    for epoch in range(200):
        training_loss = 0.0
        train_acc = val_acc = 0.0
        for X, Y in tqdm(train_loader):
            #print(X.shape, Y.shape)
            X = X.view(X.size(dim=0)*2,2,opt.imageSize,opt.imageSize)
            Y = Y.view(Y.size(dim=0)*2,1)
            #print("X, Y output:", X.shape, Y.shape, Y)
            # Y, label: 0: different, 1: same
            '''
            for i in range(opt.batchSize*2):
                for j in range(2):
                    plt.imshow(X[i, j], cmap="gray")
                    plt.savefig('test_{}_{}.png'.format(i, j))
                    plt.close()
            
            return
            '''
            X = X.cuda()
            Y = Y.cuda()
            if opt.stn:
                ori_X_size = X.size(dim=0)
                X = X.view(ori_X_size*2,1,opt.imageSize,opt.imageSize)
                X_r = STN_net(X)
                X = X_r.view(ori_X_size,2,opt.imageSize,opt.imageSize)
            pred = discriminator(X)
            loss = bce(pred, Y.float())

            optimizer.zero_grad()
            if opt.stn:
                optimizer_stn.zero_grad()
            loss.backward()
            optimizer.step()
            if opt.stn:
                optimizer_stn.step()
            training_loss += loss.data
            train_acc += get_pct_accuracy(pred, Y)

        if (epoch + 1) % 2 == 0:
            validation_loss = 0.0
            for X_val, Y_val in tqdm(test_loader):
                # validate your model
                #X_val, Y_val = loader.fetch_batch("val")
                X_val = X_val.view(X_val.size(dim=0)*2,2,opt.imageSize,opt.imageSize).cuda()
                Y_val = Y_val.view(Y_val.size(dim=0)*2,1).cuda()

                if opt.stn:
                    ori_X_val_size = X_val.size(dim=0)
                    X_val = X_val.view(ori_X_val_size*2,1,opt.imageSize,opt.imageSize)
                    X_val_r = STN_net(X_val)
                    X_val = X_val_r.view(ori_X_val_size,2,opt.imageSize,opt.imageSize)

                pred_val = discriminator(X_val)
                loss_val = bce(pred_val, Y_val.float())
                validation_loss += loss_val.data
                val_acc += get_pct_accuracy(pred_val, Y_val)

                #training_loss = loss.data[0]
                #validation_loss = loss_val.data[0]
            
            training_loss /= (float)(len(train_loader))
            validation_loss /= (float)(len(test_loader))
            train_acc /= (float)(len(train_loader))
            val_acc /= (float)(len(test_loader))

            print("Iteration: {} \t Train: Acc={}%, Loss={} \t\t Validation: Acc={}%, Loss={}".format(
            epoch, train_acc, training_loss, val_acc, validation_loss
            ))

            if best_validation_loss is None:
                best_validation_loss = validation_loss

            if best_validation_loss > (saving_threshold * validation_loss):
                print("Significantly improved validation loss from {} --> {}. Saving...".format(
                    best_validation_loss, validation_loss
                ))
                discriminator.save_to_file(os.path.join(models_path, str(epoch)+str(validation_loss)))
                if opt.stn:
                    STN_net.save_to_file(os.path.join(models_path, str(epoch)+str(validation_loss)+"_STN"))
                best_validation_loss = validation_loss
                last_saved = datetime.utcnow()

            if last_saved + save_every < datetime.utcnow():
                print("It's been too long since we last saved the model. Saving...")
                discriminator.save_to_file(os.path.join(models_path, str(validation_loss)))
                if opt.stn:
                    STN_net.save_to_file(os.path.join(models_path, str(validation_loss)+"_STN"))
                last_saved = datetime.utcnow()

def test(opt):

    if opt.cuda:
        batcher.use_cuda = True
        models.use_cuda = True

    if opt.name is None:
        # if no name is given, we generate a name from the parameters.
        # only those parameters are taken, which if changed break torch.load compatibility.
        opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                        "cuda" if opt.cuda else "cpu")

    print("Will start testing {} with parameters:\n{}\n\n".format(opt.name, opt))

    # make directory for storing models.
    models_path = os.path.join("saved_models", opt.name)

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)
    if opt.stn:
        STN_net = SpatialTransformerNetwork(I_size=(opt.imageSize, opt.imageSize),
                                            I_r_size=(opt.imageSize, opt.imageSize),
                                            I_channel_num=1)

    if opt.cuda:
        discriminator.cuda()
        if opt.stn:
            STN_net.cuda()

    # load from a previous checkpoint, if specified.
    
    discriminator.load_state_dict(torch.load(os.path.join(models_path, opt.load)))
    if opt.stn:
        STN_net.load_state_dict(torch.load(os.path.join(models_path, opt.load+"_STN")))
    
    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.cuda:
        bce = bce.cuda()

    # load the dataset in memory.
    sigdataset_test = SigDataset(opt.data, train=False, image_size=opt.imageSize)
    
    test_loader = DataLoader(sigdataset_test, batch_size=opt.batchSize, shuffle=False)
    
    val_acc = 0.0
    validation_loss = 0.0
    for X_val, Y_val in tqdm(test_loader):
        # validate your model
        X_val = X_val.view(X_val.size(dim=0)*2,2,opt.imageSize,opt.imageSize).cuda()
        Y_val = Y_val.view(Y_val.size(dim=0)*2,1).cuda()
        #print(X_val.shape)
        #print(X_val[0,0])
        save_image(X_val[1,0], 'img_0.png')
        save_image(X_val[1,1], 'img_1.png')

        if opt.stn:
            ori_X_val_size = X_val.size(dim=0)
            X_val = X_val.view(ori_X_val_size*2,1,opt.imageSize,opt.imageSize)
            X_val_r = STN_net(X_val)
            X_val = X_val_r.view(ori_X_val_size,2,opt.imageSize,opt.imageSize)
        
            #print(X_val[0,0])
            #save_image(X_val[1,0], 'img_0_r_.png')
            #save_image(X_val[1,1], 'img_1_r_.png')
            #save_image(X_val[1,0] - X_val[1,1], 'img_mi_.png')
            #save_image(X_val[0,0], 'img_0_r.png')
            #save_image(X_val[0,1], 'img_1_r.png')
            #save_image(X_val[0,0] - X_val[0,1], 'img_mi.png')
            #"""
            #print(X_val.shape)
            img_0 = X_val[0,0].detach().cpu().numpy()
            img_1 = X_val[0,1].detach().cpu().numpy()
            plt.imsave('img_0_r.png',img_0)
            plt.imsave('img_1_r.png',img_1)
            img_0 = X_val[1,0].detach().cpu().numpy()
            img_1 = X_val[1,1].detach().cpu().numpy()
            plt.imsave('img_0_r_.png',img_0)
            plt.imsave('img_1_r_.png',img_1)
            #plt.imsave('img_mi.png',img_0 - img_1)
            #"""
        
        pred_val = discriminator(X_val)
        loss_val = bce(pred_val, Y_val.float())
        validation_loss += loss_val.data
        val_acc += get_pct_accuracy(pred_val, Y_val)
        print(pred_val, Y_val)
        #"""
        ### try display arc ###
        arc = discriminator.arc

        sample = X_val[0]

        all_hidden = arc._forward(sample[None, :, :])[:, 0, :]  # (2*numGlimpses, controller_out)
        glimpse_params = torch.tanh(arc.glimpser(all_hidden))
        masks = arc.glimpse_window.get_attention_mask(glimpse_params, mask_h=opt.imageSize, mask_w=opt.imageSize)

        # separate the masks of each image.
        masks1 = []
        masks2 = []
        for i, mask in enumerate(masks):
            if i % 2 == 1:  # the first image outputs the hidden state for the next image
                masks1.append(mask)
            else:
                masks2.append(mask)

        for i, (mask1, mask2) in enumerate(zip(masks1, masks2)):
            display(opt, sample[0], mask1, sample[1], mask2, "img_{}".format(i))
        return
        #"""

    validation_loss /= (float)(len(test_loader))
    val_acc /= (float)(len(test_loader))

    print("Validation: Acc={}%, Loss={}".format(val_acc, validation_loss))


def main() -> None:
    opt = parser.parse_args()
    if not opt.test_only:
        train(opt)
    else:
        test(opt)


if __name__ == "__main__":
    main()
