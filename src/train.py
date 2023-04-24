# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Training DInPNet code

Usage: python train.py
"""
import time
import torch.nn.functional as F
from torch import optim
import torch
import numpy as np

from utils import TrainingConfiguration, SystemConfiguration, setup_system, save_model
from dataloader import get_data
from network import get_model
import config

def train(training_configuration, net, opt, train_loader, epoch_idx, loss_fn):
    """
    This function trains the net for given epoch number
    """
    # change net in training mood
    net.train()

    # to get batch loss
    batch_loss = np.array([])

    # to get batch accuracy
    batch_acc = np.array([])

    for _, (data, target) in enumerate(train_loader):

        # clone target
        indx_target = target.clone()
        # send data to device (its is medatory if GPU has to be used)
        data = data.to(training_configuration.device)
        # send target to device
        target = target.to(training_configuration.device)

        # reset parameters gradient to zero
        opt.zero_grad()

        # forward pass to the net
        output = net(data)

        # cross entropy loss
        loss = loss_fn(output, target)

        # find gradients w.r.t training parameters
        loss.backward()
        # Update parameters using gardients
        opt.step()

        batch_loss = np.append(batch_loss, [loss.item()])

        # Score to probability using softmax
        prob = F.softmax(output, dim=1)

        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]

        # correct prediction
        correct = pred.cpu().eq(indx_target).sum()

        # accuracy
        acc = float(correct) / float(len(data))

        batch_acc = np.append(batch_acc, [acc])

    epoch_loss = batch_loss.mean()
    epoch_acc = batch_acc.mean()
    print(f"Epoch: {epoch_idx} \nTrain Loss: {epoch_loss:.6f} Accuracy: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def validate(training_configuration, net, test_loader, loss_fn):
    """
    This function validate train net performance
    """
    net.eval()
    test_loss = 0.0
    count_corect_predictions = 0

    for data, target in test_loader:
        indx_target = target.clone()
        data = data.to(training_configuration.device)

        target = target.to(training_configuration.device)

        output = net(data)
        # add loss for each mini batch
        test_loss += loss_fn(output, target).item()

        # Score to probability using softmax
        prob = F.softmax(output, dim=1)

        _, pred = prob.data.max(dim=1)

        # add correct prediction count
        count_corect_predictions += pred.cpu().eq(indx_target).sum()

    # average over number of mini-batches
    test_loss = test_loss / len(test_loader)

    # average over number of dataset
    accuracy = 100. * count_corect_predictions / len(test_loader.dataset)

    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {count_corect_predictions}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n")

    return test_loss, accuracy / 100.0

def main(net, opt, lr_scheduler = None, loss=None, system_configuration = SystemConfiguration(),
         training_configuration = TrainingConfiguration()):
    """
    Train the net for given number of epochs with given parameters
    """

    # system configuration
    setup_system(system_configuration)

    # batch size
    batch_size_to_set = training_configuration.batch_size
    # num_workers
    num_workers_to_set = training_configuration.num_workers
    # epochs
    epochs = training_configuration.epochs

    # if GPU is available use training config,
    # else lowers batch_size, num_workers and epochs count
    if torch.cuda.is_available():
        print("Training using GPU")
    else:
        batch_size_to_set = 16
        num_workers_to_set = 4
        print("Training using CPU")

    # data loader1
    train_loader, test_loader = get_data(
        batch_size=batch_size_to_set,
        data_root=training_configuration.data_root,
        num_workers=num_workers_to_set
    )

    # send net to device (GPU/CPU)
    net.to(training_configuration.device)

    best_accuracy = torch.tensor(-np.inf)

    # epoch train/test loss
    epoch_train_loss = np.array([])
    epoch_test_loss = np.array([])

    # epch train/test accuracy
    epoch_train_acc = np.array([])
    epoch_test_acc = np.array([])

    # Calculate Initial Test Loss

    init_val_loss, init_val_accuracy = validate(training_configuration, net, test_loader, loss)
    print(f"Initial Test Loss : {init_val_loss:.6f}, \nInitial Test Accuracy : {init_val_accuracy*100:.3f}%\n")

    # trainig time measurement
    t_begin = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train(training_configuration, net, opt, train_loader, epoch, loss)
        epoch_train_loss = np.append(epoch_train_loss, [train_loss])
        epoch_train_acc = np.append(epoch_train_acc, [train_acc])
        elapsed_time = time.time() - t_begin
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * training_configuration.epochs - elapsed_time

        print(f"Elapsed {elapsed_time:.2f}s, {speed_epoch:.2f} s/epoch, {speed_batch:.2f} s/batch, ets {eta:.2f}s")

        if epoch % training_configuration.test_interval == 0:
            current_loss, current_accuracy = validate(training_configuration, net, test_loader, loss)
            epoch_test_loss = np.append(epoch_test_loss, [current_loss])
            epoch_test_acc = np.append(epoch_test_acc, [current_accuracy])

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy

                print('Model Improved. Saving the model...\n')
                save_model(net, device=training_configuration.device)

        # lr_scheduler step/ update learning rate
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(train_loss)
                print(f"Bad Epochs:{lr_scheduler.num_bad_epochs}")
            else:
                lr_scheduler.step()
        else:
            print(f"Learning Rate: {opt.param_groups[0]['lr']:.5f}")
    print(f"Total time: {time.time() - t_begin:.2f}, Best Accuracy: {best_accuracy:.3f}")

if __name__ == "__main__":
    # get the model
    train_config = TrainingConfiguration()

    model = get_model(num_classes = config.NUM_OF_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr = train_config.init_learning_rate)
    criterian = torch.nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.FACTOR,
        patience=config.PATIENCE,
        verbose=config.VERBOSE,
        threshold=config.THRESHOLD,
        min_lr=config.MIN_LR)

    main(model, optimizer, scheduler, criterian)
