from builtins import NotImplementedError
import os
import sys
import argparse
import torch
from torch.autograd import Variable
from torch.nn import functional as f
from datetime import datetime, timedelta
import pandas as pd

import batcher
from sig_dataloader import single_test_dataset
#from batcher import Batcher
import models
from models import ArcBinaryClassifier_conv
from torchvision.utils import save_image, make_grid
from module.transformation_simple_v3 import SpatialTransformerNetwork
from module.loss import ContrastiveLoss, ssim
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int,
                    default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the height / width of the input image to ARC')
parser.add_argument('--glimpseSize', type=int, default=8,
                    help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=128,
                    help='number of hidden states in ARC controller')
parser.add_argument('--numGlimpses', type=int, default=6,
                    help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for saving'
                                                 ' model checkpoints in a separate folder.')
parser.add_argument('--load', default=None,
                    help='the model to load from. Start fresh if not specified.')
parser.add_argument('--data', type=str,
                    default="./../ChiSig", help='data path')
parser.add_argument('--stn', action='store_true', help='stn net')
parser.add_argument('--test_only', action='store_true', help='test mode')

###STN###
parser.add_argument('--box_num', type=int, default=3, help='the # box in STN')
parser.add_argument('--boxSize', type=int, default=2, help='the X box in STN')

###ARC###
parser.add_argument('--res', action='store_true', help='ResARC')

###Loss###
parser.add_argument('--loss', type=str, default="bce",
                    help='select loss')  # ['bce', 'con']

parser.add_argument('--comment', type=str, default="", help='some note')


def get_pct_accuracy(pred: Variable, target, path=None) -> int:
    hard_pred = (pred > 0.5).int()
    #correct = (hard_pred == target).sum().data[0]
    correct = (hard_pred == target).sum().data
    accuracy = float(correct) / target.size()[0]
    """
    if accuracy != 1 and path is not None:
        f = open('wrong_result.txt', 'a')
        f.write(str(pred.tolist()))
        f.write(str(path)+'\n')
        f.close()
    """
    accuracy = int(accuracy * 100)
    return accuracy


def compute_accuracy_roc(predictions, labels, step=None):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    nsame = np.sum(labels == 1)
    ndiff = np.sum(labels == 0)
    if step is None:
        step = 0.00005

    max_acc, min_frr, min_far = 0.0, 1.0, 1.0
    min_dif = 1.0
    d_optimal = 0.0
    tpr_arr, fpr_arr, far_arr, frr_arr, d_arr = [], [], [], [], []
    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d     # pred = 1
        idx2 = predictions.ravel() > d      # pred = 0

        tpr = float(np.sum(labels[idx1] == 1)) / nsame
        tnr = float(np.sum(labels[idx2] == 0)) / ndiff

        frr = float(np.sum(labels[idx2] == 1)) / nsame
        far = float(np.sum(labels[idx1] == 0)) / ndiff

        tpr_arr.append(tpr)
        far_arr.append(far)
        frr_arr.append(frr)
        d_arr.append(d)

        acc = 0.5 * (tpr + tnr)

        # print(f"Threshold = {d} | Accuracy = {acc:.4f}")

        if acc > max_acc:
            max_acc = acc
            d_optimal = d

            # FRR, FAR metrics
            min_frr = float(np.sum(labels[idx2] == 1)) / nsame
            min_far = float(np.sum(labels[idx1] == 0)) / ndiff

        if abs(far-frr) < min_dif:
            min_dif = abs(far-frr)
            d_optimal_diff = d

            # FRR, FAR metrics
            min_dif_frr = float(np.sum(labels[idx2] == 1)) / nsame
            min_dif_far = float(np.sum(labels[idx1] == 0)) / ndiff

    print("EER: {} @{}".format((min_dif_frr+min_dif_far)/2.0, d_optimal_diff))
    metrics = {"best_acc": max_acc, "best_frr": min_frr, "best_far": min_far,
               "tpr_arr": tpr_arr, "far_arr": far_arr, "frr_arr": frr_arr, "d_arr": d_arr}
    return metrics, d_optimal


def plot_roc(tpr, fpr, fname):
    assert len(tpr) == len(fpr)
    plt.plot(fpr, tpr, marker='.')
    plt.plot(fpr, fpr, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f"./ROC_{fname}.png", dpi=300)


def find_eer(far, frr, thresholds, fname):
    plt.plot(thresholds, far, marker='o', label='far')
    plt.plot(thresholds, frr, marker='o', label='frr')
    plt.legend()
    plt.xlabel('thresh')
    plt.ylabel('far/frr')
    plt.title('find eer')
    plt.savefig(f"./EER_{fname}.png")
    plt.close()


def display(opt, image1, mask1, image2, mask2, name="hola.png"):
    _, ax = plt.subplots(1, 2)

    # a heuristic for deciding cutoff
    masking_cutoff = 0.3 / (opt.glimpseSize)**2

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


def test(opt):

    if opt.cuda:
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
    discriminator = ArcBinaryClassifier_conv(num_glimpses=opt.numGlimpses,
                                             glimpse_h=opt.glimpseSize,
                                             glimpse_w=opt.glimpseSize,
                                             controller_out=opt.numStates,
                                             res=opt.res)
    if opt.stn:
        STN_net = SpatialTransformerNetwork(I_size=(opt.imageSize, opt.imageSize),
                                            I_r_size=(opt.imageSize //
                                                      2, opt.imageSize//2),
                                            I_channel_num=1,
                                            box_num=opt.box_num,
                                            boxSize=opt.boxSize)

    if opt.cuda:
        discriminator.cuda()
        if opt.stn:
            STN_net.cuda()
    # '''
    # load from a previous checkpoint, if specified.
    discriminator.load_state_dict(torch.load(
        os.path.join(models_path, opt.load)))
    if opt.stn:
        STN_net.load_state_dict(torch.load(
            os.path.join(models_path, opt.load+"_STN")))
    # '''
    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.cuda:
        bce = bce.cuda()

    # load the dataset in memory.
    #sigdataset_test = SigDataset(opt.data, train=False, image_size=opt.imageSize)
    ###
    # '''
    path_dict = {'Anchor': ['./../BHSig260_demo/Hindi/003/H-S-3-G-04.tif'], 'positive': ['./../BHSig260_demo/Hindi/003/H-S-3-G-22.tif'], 'negative': ['./../BHSig260_demo/Hindi/003/H-S-3-F-12.tif']} #{'Anchor': ['./../BHSig260_demo/Bengali/003/B-S-3-G-14.tif'], 'positive': ['./../BHSig260_demo/Bengali/003/B-S-3-G-23.tif'], 'negative': ['./../BHSig260_demo/Bengali/003/B-S-3-F-20.tif']}
    sigdataset_test = single_test_dataset(path_dict['Anchor'][0], path_dict['positive'][0], path_dict['negative'][0], image_size=opt.imageSize, mode = 'cropped')

    test_loader = DataLoader(
        sigdataset_test, batch_size=opt.batchSize, shuffle=False)
    # '''
    # return
    val_acc = 0.0
    validation_loss = 0.0
    np_loss = np.zeros(shape=(0,))
    gt_loss = np.zeros(shape=(0,))
    for X_val, Y_val, path in tqdm(test_loader):
        if opt.stn:
            STN_net.eval()
        discriminator.eval()
        # validate your model
        X_val = X_val.view(-1, 2,
                           opt.imageSize, opt.imageSize).cuda()
        Y_val = Y_val.view(-1, 1).cuda()
        #save_image(X_val[1,0], 'img_0.png')
        #save_image(X_val[1,1], 'img_1.png')
        # """
        fig, ax = plt.subplots(2, X_val.size(dim=0))
        fig.set_figheight(3)
        fig.set_figwidth(3)
        for i in range(X_val.size(dim=0)):
            # Display the image
            ax[1][i].imshow(X_val[i, 0].detach().cpu().numpy(),
                            cmap=mpl.cm.bone)
            ax[0][i].imshow(X_val[i, 1].detach().cpu().numpy(),
                            cmap=mpl.cm.bone)
            print(i, ":", ssim(
                X_val[i, 0, None, None], X_val[i, 1, None, None]))
        plt.savefig("img")
        plt.close()
        # """
        if opt.stn:
            ori_X_val_size = X_val.size(dim=0)
            X_val = X_val.view(ori_X_val_size*2, 1,
                               opt.imageSize, opt.imageSize)
            X_val_r = STN_net(X_val)

            X_val = torch.reshape(
                X_val_r, (ori_X_val_size*opt.box_num, 2, opt.imageSize//2, opt.imageSize//2))
            # print(X_val.shape)
            #grid = make_grid(torch.unsqueeze(X_val.permute(1, 0, 2, 3)[1], 1), nrow=ori_X_val_size*opt.box_num)
            #save_image(grid, 'grid_demo.png')
            """
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
            """
            # """
            fig, ax = plt.subplots(2, X_val.size(dim=0))
            fig.set_figheight(3)
            fig.set_figwidth(6)
            for i in range(X_val.size(dim=0)):
                # Display the image
                ax[1][i].imshow(
                    X_val[i, 0].detach().cpu().numpy(), cmap=mpl.cm.bone)
                ax[0][i].imshow(
                    X_val[i, 1].detach().cpu().numpy(), cmap=mpl.cm.bone)
                print(i, ":", ssim(
                    X_val[i, 0, None, None], X_val[i, 1, None, None]))
            plt.savefig("img_stn")
            plt.close()
            # """

        pred_val, embedding = discriminator(X_val)
        print(pred_val)

        pred_val = torch.reshape(pred_val, (-1, Y_val.size(dim=0)))
        pred_val = torch.mean(pred_val, dim=0).unsqueeze(1)

        loss_val = bce(pred_val, Y_val.float())
        validation_loss += loss_val.data
        np_loss = np.append(np_loss, pred_val.cpu().detach().numpy())
        gt_loss = np.append(gt_loss, Y_val.cpu().detach().numpy())
        val_acc += get_pct_accuracy(pred_val, Y_val, path=path)
        print(pred_val, Y_val)
        '''
        arc = discriminator.arc
        featuremaps_pairs = arc.view_feature(X_val)
        print(featuremaps_pairs[0][0].shape)
        featur_i_want = featuremaps_pairs[0][0]
        for i in range(featur_i_want.size()[0]):
            img_0 = featur_i_want[i].squeeze().detach().cpu().numpy()
            plt.imsave('feature/feature_{}.png'.format(i), img_0)
        f_X = torch.reshape(featur_i_want, (64, -1))
        print(f_X.shape)
        ff_X = torch.matmul(f_X, torch.transpose(f_X, 0, 1))
        ff_X = torch.matmul(ff_X, f_X)
        ff_X = torch.reshape(ff_X, (64, opt.imageSize//2, opt.imageSize//2))
        for i in range(ff_X.size()[0]):
            img_0 = ff_X[i].squeeze().detach().cpu().numpy()
            plt.imsave('feature/feature_a_{}.png'.format(i), img_0)
        '''
        # """
        for item in range(X_val.size(dim=0)):
            ### try display arc ###
            arc = discriminator.arc

            sample = X_val[item]

            all_hidden, _ = arc._forward(sample[None, :, :])
            all_hidden = all_hidden[:, 0, :]  # (2*numGlimpses, controller_out)
            glimpse_params = torch.tanh(arc.glimpser(all_hidden))
            centers_y, centers_x, deltas_y, deltas_x = arc.glimpse_window.get_attention_box(
                glimpse_params, mask_h=opt.imageSize//opt.boxSize, mask_w=opt.imageSize//opt.boxSize)
            # print(centers_x, centers_y, deltas_x, deltas_y)
            gsize = int(opt.imageSize//opt.boxSize)

            fig, ax = plt.subplots(2, opt.numGlimpses)
            fig.set_figheight(2)
            fig.set_figwidth(10)
            for i in range(opt.numGlimpses):
                # Display the image
                ax[0][i].imshow(sample[1].data.cpu().numpy(), cmap=mpl.cm.bone)
                rect = mpl.patches.Rectangle((centers_x[2*i] - 0.5*deltas_x[2*i], centers_y[2*i] - 0.5*deltas_y[2*i]),
                                             deltas_x[2*i], deltas_y[2*i], linewidth=1, edgecolor='r', facecolor='none')
                ax[0][i].add_patch(rect)

                ax[1][i].imshow(sample[0].data.cpu().numpy(), cmap=mpl.cm.bone)
                rect = mpl.patches.Rectangle((centers_x[2*i+1] - 0.5*deltas_x[2*i+1], centers_y[2*i+1] - 0.5 *
                                              deltas_y[2*i+1]), deltas_x[2*i+1], deltas_y[2*i+1], linewidth=1, edgecolor='r', facecolor='none')
                ax[1][i].add_patch(rect)

            images_path = os.path.join("visualization", opt.name)
            os.makedirs(images_path, exist_ok=True)
            plt.savefig(os.path.join(images_path, "all_{}".format(item)))
            plt.close()

            masks = arc.glimpse_window.get_attention_mask(
                glimpse_params, mask_h=opt.imageSize//opt.boxSize, mask_w=opt.imageSize//opt.boxSize)

            # separate the masks of each image.
            masks1 = []
            masks2 = []
            for i, mask in enumerate(masks):
                if i % 2 == 1:  # the first image outputs the hidden state for the next image
                    masks1.append(mask)
                else:
                    masks2.append(mask)

            for i, (mask1, mask2) in enumerate(zip(masks1, masks2)):
                display(opt, sample[0], mask1, sample[1],
                        mask2, "img_{}_{}".format(item, i))
            # return
        # """
        return

    validation_loss /= (float)(len(test_loader))
    val_acc /= (float)(len(test_loader))

    print("Validation: Acc={}%, Loss={}".format(val_acc, validation_loss))

    # evaluate EER
    np_loss = 1 - np_loss
    #gt_loss = 1- gt_loss
    #print(np_loss, gt_loss)
    metrics, thresh_optimal = compute_accuracy_roc(np_loss, gt_loss, step=5e-5)
    data_df = pd.DataFrame({"dist": np_loss, "y_true": gt_loss})
    data_gb = data_df.groupby("y_true")
    pos_dist = data_gb.get_group(1)["dist"]
    neg_dist = data_gb.get_group(0)["dist"]
    plt.hist(np.array(pos_dist), 200, facecolor='g', alpha=0.3)
    plt.hist(np.array(neg_dist), 200, facecolor='r', alpha=0.3)
    plt.savefig(f"./density_Chi.png")
    plt.close()

    print("d optimal: {}".format(thresh_optimal))
    print("Metrics obtained: \n" + '-'*50)
    print(f"Acc: {metrics['best_acc'] * 100 :.4f} %")
    print(f"FAR: {metrics['best_far'] * 100 :.4f} %")
    print(f"FRR: {metrics['best_frr'] * 100 :.4f} %")
    print('-'*50)

    find_eer(metrics['far_arr'], metrics['frr_arr'],
             metrics['d_arr'], "test_Chi")
    plot_roc(np.array(metrics['tpr_arr']),
             np.array(metrics['far_arr']), "test_Chi")


def main() -> None:
    opt = parser.parse_args()
    test(opt)


if __name__ == "__main__":
    main()
