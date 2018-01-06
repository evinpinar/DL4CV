from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        num_iterations = num_epochs*iter_per_epoch
        best_val_acc = -1
        loss_func = self.loss_func
        best_params = {}
        
        train_acc = 0;
        val_acc = 0;

        for epoch in range(num_epochs):

            model.train()
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                if model.is_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                optim.zero_grad()
                outputs = model(inputs)
                train_loss = loss_func(outputs, labels)
                loss.backward()
                optim.step()

                self.train_loss_history.append(train_loss.data[0])

                t = epoch*(iter_per_epoch-1)+i
                if t % log_nth == 0 :
                    print('[Iteration %d / %d] TRAIN loss: %f' % (t, num_iterations, train_loss.data[0]))

                if epoch == len(train_loader)
                    y_pred = np.argmax(outputs.data.numpy(), axis = 1)
                    train_acc = np.mean(outputs == labels)
                    self.train_acc_history.append(train_acc)

            model.eval()
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                if model.is_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                val_pred = model.forward(X_val)
                val_loss = loss_func(val_pred, y_val)

                self.val_loss_history.append(val_loss.data[0])

                if i == len(val_loader)
                    y_pred = np.argmax(outputs.data.numpy(), axis = 1)
                    val_acc = np.mean(outputs == labels)
                    self.train_acc_history.append(val_acc)
            
            print('[Epoch %d / %d] train acc: %f; val_acc: %f' % (epoch, num_epochs, train_acc, val_acc))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
