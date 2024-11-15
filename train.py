import logging
import argparse
import math
import os
import sys
import random
import numpy
import yaml
from sklearn import metrics
from time import strftime, localtime

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import Tokenizer4Bert, ABSADataset
from models.emofuse import AdaptWin
from torch.nn import functional as F

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        # 根据model类型选择embedded输入
        tokenizer = Tokenizer4Bert(opt['max_seq_len'], opt['pretrained_bert_name'])
        bert = BertModel.from_pretrained(opt['pretrained_bert_name'])
        self.model = opt['model_class'](bert, opt).to(opt['device'])

        self.trainset = ABSADataset(opt['dataset_file']['train'], tokenizer)
        self.testset = ABSADataset(opt['dataset_file']['test'], tokenizer)
        assert 0 <= opt['valset_ratio'] < 1
        if opt['valset_ratio'] > 0:
            valset_len = int(len(self.trainset) * opt['valset_ratio'])
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt['device'].type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt['device'].index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in self.opt:
            logger.info('>>> {0}: {1}'.format(arg, self.opt[arg]))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt['initializer'](p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    
    def _train_vae(self, criterion, optimizer, train_data_loader):
        self.model.train()
        for epoch in range(self.opt['vae_epochs']):
            total_loss = 0
            for batch in train_data_loader:
                optimizer.zero_grad()
                inputs = [batch[col].to(self.opt['device']) for col in self.opt['inputs_cols']]
                # Forward pass through the model in VAE mode
                recon_x, mu, logvar, _ = self.model(inputs, vae_only=True)
                # Get the target embeddings
                target_embeddings = self.model.dropout(
                    self.model.bert_spc.embeddings.word_embeddings(inputs[0])
                )
                # Compute the reconstruction loss
                recon_loss = F.mse_loss(recon_x, target_embeddings, reduction='sum')
                # Compute the KL divergence
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                # Total VAE loss
                loss = recon_loss + self.opt['beta'] * kld_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_data_loader.dataset)
            logger.info('VAE Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.opt['vae_epochs'], avg_loss))



    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt['num_epoch']):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                optimizer.zero_grad()
                inputs = [batch[col].to(self.opt['device']) for col in self.opt['inputs_cols']]
                # outputs = self.model(inputs)
                outputs, representations = self.model(inputs)
                targets = batch['polarity'].to(self.opt['device'])
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt['log_step'] == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_{2}_val_acc_{3}_f1_{4}'.format(self.opt['model_name'], self.opt['dataset'], self.opt['threshold'], round(val_acc, 4), round(val_f1, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
            logger.info('> best_val_acc: {:.4f}, val_f1: {:.4f}'.format(max_val_acc, max_val_f1))
            # Early stopping logic
            if self.opt['early_stop']:
                if i_epoch - max_val_epoch >= self.opt['patience']:
                    logger.info('>> early stop.')
                    break
            # if i_epoch - max_val_epoch >= self.opt['patience']:
            #     print('>> early stop.')
            #     break

        return path

    def _evaluate_acc_f1(self, data_loader, _istest=False):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt['device']) for col in self.opt['inputs_cols'][:len(self.opt['inputs_cols'])]]
                t_targets = t_batch['polarity'].to(self.opt['device'])
                # t_outputs = self.model(t_inputs)
                t_outputs, representations = self.model(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # VAE Training
        vae_params = list(self.model.vae.parameters())
        vae_optimizer = torch.optim.Adam(vae_params, lr=self.opt['vae_lr'])
        vae_criterion = nn.MSELoss(reduction='sum')
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt['batch_size'], shuffle=True)
        
        logger.info('Starting VAE Training...')
        self._train_vae(vae_criterion, vae_optimizer, train_data_loader)
        logger.info('VAE Training Completed.')

        # Freeze VAE parameters (optional, depending on your strategy)
        for param in self.model.vae.parameters():
            param.requires_grad = False

        # Classifier Training
        classifier_params = filter(
            lambda p: p.requires_grad and id(p) not in list(map(id, vae_params)),
            self.model.parameters()
        )
        optimizer = self.opt['optimizer'](classifier_params, lr=self.opt['lr'], weight_decay=self.opt['l2reg'])
        criterion = nn.CrossEntropyLoss()
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt['batch_size'], shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt['batch_size'], shuffle=False)
        
        logger.info('Starting Classifier Training...')
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader, True)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))



def main():
    with open('config.yaml', 'r', encoding='utf-8') as file:
        opt = yaml.safe_load(file)
    print("opt: ", opt)

    if opt['seed'] is not None:
        random.seed(opt['seed'])
        numpy.random.seed(opt['seed'])
        torch.manual_seed(opt['seed'])
        torch.cuda.manual_seed(opt['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt['seed'])
    
    if 'early_stop' not in opt:
        opt['early_stop'] = True  # Default to True

    model_classes = {
        'AdaptWin_bert': AdaptWin,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    input_colses = {
        'AOAN_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices','aspect_boundary'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    opt['model_class'] = model_classes[opt['model_name']]
    opt['dataset_file'] = dataset_files[opt['dataset']]
    opt['inputs_cols'] = input_colses[opt['model_name']]
    opt['initializer'] = initializers[opt['initializer']]
    opt['optimizer'] = optimizers[opt['optimizer']]
    opt['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt['device'] is None else torch.device(opt['device'])

    # Ensure dropout is added to the options dictionary
    if 'dropout' not in opt:
        opt['dropout'] = 0.5  # Default value, adjust as needed

    log_file = '{}-{}-{}.log'.format(opt['model_name'], opt['dataset'], strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
