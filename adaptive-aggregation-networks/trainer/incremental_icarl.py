##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## Modified from: https://github.com/hshustc/CVPR19_Incremental_Learning
## Max Planck Institute for Informatics
## yaoyao.liu@mpi-inf.mpg.de
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Training code for iCaRL """
import torch
import tqdm
import numpy as np
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from utils.misc import *
from utils.process_fp import process_inputs_fp
from utils.ebm_aligner import EBMAligner
import torch.nn.functional as F

def incremental_train_and_eval(the_args, epochs, fusion_vars, ref_fusion_vars, b1_model, ref_model, b2_model, ref_b2_model, tg_optimizer, tg_lr_scheduler, fusion_optimizer, fusion_lr_scheduler, trainloader, testloader, new_data_trainloader, iteration, start_iteration, X_protoset_cumuls, Y_protoset_cumuls, order_list,lamda, dist, K, lw_mr, balancedloader, T=None, beta=None, aligner=None, prev_valid_loader=None, fix_bn=False, weight_per_class=None, device=None):

    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set the 1st branch reference model to the evaluation mode
    ref_model.eval()

    # Get the number of old classes
    num_old_classes = ref_model.fc.out_features

    # If the 2nd branch reference is not None, set it to the evaluation mode
    if iteration > start_iteration+1:
        ref_b2_model.eval()

    if the_args.enable_ebm:
        aligner = EBMAligner()

    for epoch in range(epochs):
        # Learn the EBM
        if aligner is not None and epoch == the_args.ebm_start_epoch:
            aligner.learn_ebm(ref_model, b1_model, new_data_trainloader, prev_valid_loader)
        elif aligner is not None and epoch % the_args.ebm_update_freq == 0 and epoch != 0:
            aligner.learn_ebm(ref_model, b1_model, new_data_trainloader, prev_valid_loader)

        # Start training for the current phase, set the two branch models to the training mode
        b1_model.train()
        b2_model.train()

        # Fix the batch norm parameters according to the config
        if fix_bn:
            for m in b1_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Set all the losses to zeros
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_ebm_loss = 0
        train_ebm_distiller_loss = 0

        # Set the counters to zeros
        correct = 0
        total = 0

        # Learning rate decay
        tg_lr_scheduler.step()
        fusion_lr_scheduler.step()

        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])

        for batch_idx, (inputs, targets) in enumerate(trainloader):

            # Get a batch of training samples, transfer them to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Clear the gradient of the paramaters for the tg_optimizer
            tg_optimizer.zero_grad()

            # Forward the samples in the deep networks
            outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
            _, z = b1_model(inputs, return_z_also=True)

            if iteration == start_iteration+1:
                ref_outputs = ref_model(inputs)
            else:
                ref_outputs, ref_features_new = process_inputs_fp(the_args, ref_fusion_vars, ref_model, ref_b2_model, inputs)
            # Loss 1: feature-level distillation loss
            loss1 = nn.KLDivLoss()(F.log_softmax(outputs[:,:num_old_classes]/T, dim=1), \
                F.softmax(ref_outputs.detach()/T, dim=1)) * T * T * beta * num_old_classes

            # Loss 2: classification loss
            loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

            # Loss 3: ebm loss (latent alignment)
            loss3 = torch.zeros(1).to(device)
            if aligner is not None and aligner.is_enabled and the_args.enable_ebm_aligner:
                loss3 = aligner.loss(z)

            # Loss 4: ebm loss (latent distiller)
            loss4 = torch.zeros(1).to(device)
            if aligner is not None and aligner.is_enabled and the_args.enable_ebm_distiller:
                aligned_z = aligner.align_latents(z).clone().detach()
                loss4 = nn.KLDivLoss()(F.log_softmax(outputs[:, :num_old_classes] / T, dim=1),
                                       F.softmax(ref_model.fc(aligned_z).detach() / T, dim=1)) * T * T * beta * num_old_classes

            # Sum up all looses
            loss = loss1 + loss2 + loss3 + loss4

            # Backward and update the parameters
            loss.backward()
            tg_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_ebm_loss += loss3.item()
            train_ebm_distiller_loss += loss4.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print the training losses and accuracies
        print('Train set: {}, train loss1: {:.4f}, train loss2: {:.4f}, train ebm align loss: {:.4f}, '
              'train ebm distiller loss: {:.4f}, train loss: {:.4f} accuracy: {:.4f}'
              .format(len(trainloader), train_loss1/(batch_idx+1), train_loss2/(batch_idx+1),
                      train_ebm_loss/(batch_idx+1), train_ebm_distiller_loss/(batch_idx+1),
                      train_loss/(batch_idx+1), 100.*correct/total))
        
        # Update the aggregation weights
        b1_model.eval()
        b2_model.eval()
        
        for batch_idx, (inputs, targets) in enumerate(balancedloader):
            fusion_optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
            loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            loss.backward()
            fusion_optimizer.step()

        # Running the test for this epoch
        b1_model.eval()
        b2_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = process_inputs_fp(the_args, fusion_vars, b1_model, b2_model, inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print('Test set: {} test loss: {:.4f} accuracy: {:.4f}'.format(len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    print("Removing register forward hook")
    return b1_model, b2_model, aligner
