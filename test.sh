#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python sig_cc_train.py --cuda --name C_11 --batchSize 4 --data ./../ChiSig --lr 1e-6 --numGlimpses 8 --imageSize 64 --numStates 1024 --glimpseSize 2 --load "tensor(0.1563, device='cuda:0')" --comment "_11: cropped, w pretrain"
CUDA_VISIBLE_DEVICES=1 python sig_cc_train.py --cuda --name B_11_0 --batchSize 4 --data ./../BHSig260/Bengali --lr 1e-6 --numGlimpses 8 --imageSize 64 --numStates 1024 --glimpseSize 2 --comment "_11: cropped, wo pretrain"
CUDA_VISIBLE_DEVICES=1 python sig_cc_train.py --cuda --name H_11_0 --batchSize 4 --data ./../BHSig260/Hindi --lr 1e-6 --numGlimpses 8 --imageSize 64 --numStates 1024 --glimpseSize 2 --comment "_11: cropped, wo pretrain"
CUDA_VISIBLE_DEVICES=1 python sig_cc_train.py --cuda --name C_11_0 --batchSize 4 --data ./../ChiSig --lr 1e-6 --numGlimpses 8 --imageSize 64 --numStates 1024 --glimpseSize 2 --comment "_11: cropped, wo pretrain"
