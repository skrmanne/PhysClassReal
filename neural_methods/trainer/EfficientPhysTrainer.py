"""Trainer for EfficientPhys."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.loss.PSD_MSELoss import PSD_MSE
from neural_methods.loss.ClassificationLoss import ClassLoss
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm

from evaluation.post_process import get_signal

class EfficientPhysTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.EFFICIENTPHYS.FRAME_DEPTH
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.num_train_batches = len(data_loader["train"])
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        self.mode = config.MODEL.MODE    # Regression or Classification mode
        self.model = EfficientPhys(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.H, mode=self.mode).to(
            self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

        # configurable loss function
        self.criterion = torch.nn.MSELoss()
        if config.MODEL.LOSS == 'psd_mse':
            if config.TEST.DATA.DATASET == 'AIR' or config.TEST.DATA.DATASET == 'AIRFLOW':
                low, high = 0.3, 0.8
            else:
                low, high = 0.08, 0.5
                # TODO: Create multiple instances of the loss to reflect the fps of each dataloader.
            self.criterion = PSD_MSE(fs=config.TEST.DATA.FS, high_pass=high, low_pass=low)
        
        if self.mode == "classification":
            self.criterion = ClassLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)

        # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []

            self.model.train()
            
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data = torch.cat((data, last_frame), 0)
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                if isinstance(self.criterion, PSD_MSE) or isinstance(self.criterion, ClassLoss):
                    # if the loss is PSD_MSE or ClassLoss, the batches should be individually run for the loss computation
                    pred_ppg = pred_ppg.view(N, -1)
                    labels = labels.view(N, -1)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item()
                
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")

        precision, recall, accuracy = [], [], []
        valid_loss = []
        self.model.eval()
        valid_step = 0
        
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):

                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_valid[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data_valid = torch.cat((data_valid, last_frame), 0)
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)

                if isinstance(self.criterion, PSD_MSE) or isinstance(self.criterion, ClassLoss):
                    # if the loss is PSD_MSE, the batches should be individually run for the loss computation
                    pred_ppg_valid = pred_ppg_valid.view(N, -1)
                    labels_valid = labels_valid.view(N, -1)
                
                loss = self.criterion(pred_ppg_valid, labels_valid)
                # if mode==classification compute precision, recall and accuracy for the predictions
                if self.mode == "classification":
                    pred_ppg_valid = torch.sigmoid(pred_ppg_valid)
                    class_preds = pred_ppg_valid >= 0.5
                    #print(class_preds[0]) #, labels_valid)
                    # compute precision, recall and accuracy bw pred_ppg_valid and labels_valid
                    prec = torch.sum(class_preds * labels_valid) / (torch.sum(class_preds) + 1e-6)
                    rec = torch.sum(class_preds * labels_valid) / (torch.sum(labels_valid) + 1e-6)
                    acc = torch.sum(class_preds == labels_valid) / (class_preds.shape[0] * class_preds.shape[1])
                    precision.append(prec.item())
                    recall.append(rec.item())
                    accuracy.append(acc.item())

                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())

            if self.mode == "classification":
                print("Validation Precision, Recall, Accuracy: ", 
                np.mean(np.asarray(precision)), np.mean(np.asarray(recall)), np.mean(np.asarray(accuracy)))
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()

        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)

                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]

                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_test[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data_test = torch.cat((data_test, last_frame), 0)
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)

                # convert to signal from class labels
                if self.mode == "classification":
                    pred_ppg_test = torch.sigmoid(pred_ppg_test)
                    pred_ppg_test = pred_ppg_test.view(N, -1)
                    pred_ppg_test = get_signal(pred_ppg_test)
                    pred_ppg_test = pred_ppg_test.view(-1, 1)

                    # convert labels as well since data is already converted to labels
                    labels_test = labels_test.view(N, -1)
                    labels_test = get_signal(labels_test)
                    labels_test = labels_test.view(-1, 1)

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
