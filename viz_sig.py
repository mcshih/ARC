import os
import argparse
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sig_dataloader import SigDataset
from models import ArcBinaryClassifier
from module.transformation import TPS_SpatialTransformerNetwork
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to ARC')
parser.add_argument('--glimpseSize', type=int, default=8, help='the height / width of glimpse seen by ARC')
parser.add_argument('--numStates', type=int, default=128, help='number of hidden states in ARC controller')
parser.add_argument('--numGlimpses', type=int, default=6, help='the number glimpses of each image in pair seen by ARC')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--name', default=None, help='Custom name for this configuration. Needed for loading model'
                                                 'and saving images')
parser.add_argument('--load', required=True, help='the model to load from.')
parser.add_argument('--data', type=str, default="./../ChiSig", help='data path')
parser.add_argument('--same', action='store_true', help='whether to generate same character pairs or not')

opt = parser.parse_args()

device = torch.device("cuda")

if opt.name is None:
    # if no name is given, we generate a name from the parameters.
    # only those parameters are taken, which if changed break torch.load compatibility.
    opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates,
                                    "cuda" if opt.cuda else "cpu")

# make directory for storing images.
images_path = os.path.join("visualization", opt.name)
os.makedirs(images_path, exist_ok=True)


# initialise the batcher
#batcher = Batcher(batch_size=opt.batchSize)
sigdataset_test = SigDataset(opt.data, train=False, image_size=opt.imageSize)
test_loader = DataLoader(sigdataset_test, batch_size=opt.batchSize, shuffle=False)

def getpatches(opt, x_arr):
    ptsz = opt.imageSize//2
    patches = []
    Batch_size ,C, H, W = x_arr.shape # 1, 256, 256
    num_H = 2
    num_W = 2

    for i in range(num_H):
        for j in range(num_W):
            start_x = i*ptsz
            end_x = start_x + ptsz
            start_y = j*ptsz
            end_y = start_y + ptsz

            patch = x_arr[:,:, start_x:end_x, start_y:end_y]
            patches.append(torch.unsqueeze(patch, 0))

    return torch.squeeze(torch.cat(patches, dim=0), 1)

def get_pct_accuracy(pred: Variable, target) -> int:
    hard_pred = (pred > 0.5).int()
    #correct = (hard_pred == target).sum().data[0]
    correct = (hard_pred == target).sum().data
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy, correct

def display(image1, mask1, image2, mask2, name="hola.png"):
    _, ax = plt.subplots(1, 2)

    # a heuristic for deciding cutoff
    masking_cutoff = 2.4 / (opt.glimpseSize)**2

    mask1 = (mask1 > masking_cutoff).data.numpy()
    mask1 = np.ma.masked_where(mask1 == 0, mask1)

    mask2 = (mask2 > masking_cutoff).data.numpy()
    mask2 = np.ma.masked_where(mask2 == 0, mask2)

    ax[0].imshow(image1.data.numpy(), cmap=mpl.cm.bone)
    ax[0].imshow(mask1, interpolation="nearest", cmap=mpl.cm.jet_r, alpha=0.7)

    ax[1].imshow(image2.data.numpy(), cmap=mpl.cm.bone)
    ax[1].imshow(mask2, interpolation="nearest", cmap=mpl.cm.ocean, alpha=0.7)

    plt.savefig(os.path.join(images_path, name))


def get_sample(discriminator):

    # size of the set to choose sample from from
    sample_size = 30
    #X, Y = batcher.fetch_batch("train", batch_size=sample_size)
    pred = discriminator(X)

    if opt.same:
        same_pred = pred[sample_size // 2:].data.numpy()[:, 0]
        mx = same_pred.argsort()[len(same_pred) // 2]  # choose the sample with median confidence
        index = mx + sample_size // 2
    else:
        diff_pred = pred[:sample_size // 2].data.numpy()[:, 0]
        mx = diff_pred.argsort()[len(diff_pred) // 2]  # choose the sample with median confidence
        index = mx

    return X[index]


def visualize():

    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        controller_out=opt.numStates)
    STN_net = TPS_SpatialTransformerNetwork(F=20,
                                            I_size=(opt.imageSize, opt.imageSize),
                                            I_r_size=(opt.imageSize, opt.imageSize),
                                            I_channel_num=1)
    discriminator.to(device)
    STN_net.to(device)
    discriminator.load_state_dict(torch.load(os.path.join("saved_models", opt.name, opt.load)))
    STN_net.load_state_dict(torch.load(os.path.join("saved_models", opt.name, opt.load+"_STN")))

    discriminator.eval()
    STN_net.eval()

    bce = torch.nn.BCELoss()
    validation_loss = 0.0
    val_acc = 0.0

    for X_val, Y_val in tqdm(test_loader):
        # validate your model
        X_val = X_val.view(X_val.size(dim=0)*2,2,opt.imageSize,opt.imageSize).to(device)
        Y_val = Y_val.view(Y_val.size(dim=0)*2,1).to(device)

        ori_X_val_size = X_val.size(dim=0)
        X_val = X_val.view(ori_X_val_size*2,1,opt.imageSize,opt.imageSize)
        X_val_r = STN_net(X_val)
        X_val_r = getpatches(opt, X_val_r)
        X_val = X_val_r.view(ori_X_val_size*4,2,opt.imageSize//2,opt.imageSize//2)

        pred_val = discriminator(X_val)

        pred_val = torch.mean(torch.reshape(pred_val, (4, -1)), dim=0)
        pred_val = torch.reshape(pred_val, Y_val.shape)

        loss_val = bce(pred_val, Y_val.float())
        validation_loss += loss_val.data
        val_acc += get_pct_accuracy(pred_val, Y_val)[1]
        
    validation_loss /= (float)(len(test_loader))
    val_acc /= (float)(2 * len(sigdataset_test))
    
    print((float)(val_acc), (float)(validation_loss))
    """
    arc = discriminator.arc

    sample = get_sample(discriminator)

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
        display(sample[0], mask1, sample[1], mask2, "img_{}".format(i))
    """


if __name__ == "__main__":
    visualize()
