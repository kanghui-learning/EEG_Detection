import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.metrics import dtw
import argparse
import logging
import sys
from datetime import datetime
import torch
import numpy as np
from utils.utils import _logger
from models.CNN import CNNModel
from models.SimpleEEGNet import SimpleEEGNet
from models.LSTM import LSTMModel
from models.CNNLSTM import CNNLSTM
from models.DeepConvNET import DeepConvNet
from models.ResNet1D import ResNet1D
from models.FCN import FCNModel
from models.STGCN import STGCN  
from models.AGCRN import AGCRN
from models.LVM import LVMModel
from models.BiGRU import BiGRUModel
from models.XGBoost import XGBoostModel
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from uuid import uuid4
from collections import Counter
from tqdm import tqdm


data_dict = {

}

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='EEG Classification'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--method', type=str, required=True, help='choose the baseline models')
    parser.add_argument('--seed', type=int, required=True,
                        help='Set the seed for the experiment')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--adj_type',type=str, help="choose the method to calculate the adj for STGCN model total is ['train','all','seperate']")
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--smooth', action='store_true', help='Use smoothed data')
    parser.add_argument('--denoise', action='store_true', help='Use denoised data')
    return parser

def evaluate(model, loader, device, A_hat=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(A_hat, inputs) if A_hat is not None else model(inputs)
            probs = outputs
            _, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds)  # Sensitivity

    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = recall
        specificity = np.mean([cm[i,i]/np.sum(cm[i,:]) if np.sum(cm[i,:]) > 0 else 0 
                               for i in range(len(cm))])

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5

    return accuracy, f1, precision, recall, sensitivity, specificity, auc

def get_adj(data):

    reshaped_data = data.reshape(-1, data.shape[1]).cpu().numpy()
    correlation_matrix = np.corrcoef(reshaped_data, rowvar=False)
    threshold = 0.5
    adjacency_matrix = np.where(np.abs(correlation_matrix) >= threshold, 1, 0)
    np.fill_diagonal(adjacency_matrix, 1)
    print("adjacency matrix",adjacency_matrix.shape)
    return adjacency_matrix

def normalize_adj(adj):
    rowsum = adj.sum(1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    return normalized_adj


if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()
    
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    logs_save_dir = args.method + '_results'
    experiment_log_dir = os.path.join('./results/', logs_save_dir, args.dataset)
    os.makedirs(experiment_log_dir, exist_ok=True)
    
    # Create results file path
    results_file = os.path.join('./results/', f'training_results_{args.dataset}.txt')
    
    # logging 
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {args.dataset}')
    logger.debug("=" * 45)
    logger.debug("Hyperparameters:")
    logger.debug(f"Method: {args.method}")
    logger.debug(f"Seed: {args.seed}")
    logger.debug(f"GPU: {args.gpu}")
    logger.debug(f"Batch Size: {args.batch_size}")
    logger.debug(f"Learning Rate: {args.learning_rate}")
    logger.debug(f"Number of Epochs: {args.num_epochs}")
    logger.debug(f"Smooth: {args.smooth}")
    logger.debug(f"Denoise: {args.denoise}")
    if args.method == 'STGCN':
        logger.debug(f"Adjacency Type: {args.adj_type}")
    logger.debug("=" * 45)

    if args.smooth and not args.denoise:
        args.dataset = args.dataset + '_smooth'
    elif args.denoise and not args.smooth:
        args.dataset = args.dataset + '_denoise'
    elif args.smooth and args.denoise:
        args.dataset = args.dataset + '_smooth_denoise'
    dataset = np.load(data_dict[args.dataset])
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    print("y_val",y_val)
    print("y_train_count",Counter(y_train))
    print("y_val_count",Counter(y_val))
    print("y_test_count",Counter(y_test))
 
    print('train size', y_train.shape, 'number 1 in train', y_train.sum())
    print('val size', y_val.shape, 'number 1 in val', y_val.sum())
    print('test size', y_test.shape, 'number 1 in test', y_test.sum())
  

    logger.debug("Data loaded----------")
    print("X_train shape", X_train.shape)
    print('X_val shape', X_val.shape)
    print('X_test shape', X_test.shape)
    
    logger.debug(f"Seed: {args.seed}")
    # hyperparameters
    batch_size = args.batch_size
    n_classes = len(np.unique(y_train, axis=0))
    print(f'n_classes: {n_classes}')
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    device = torch.device(f'cuda:{args.gpu}' if int(args.gpu) >= 0 else 'cpu')

    # dataset
    y_train, y_val, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_val, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
    
    if args.method == 'AGCRN':
        # reshape
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_train = X_train.transpose(1, 2)  # (N, num_nodes, seq_len)
        X_val = X_val.transpose(1, 2)
        X_test = X_test.transpose(1, 2)
        X_train = X_train.permute(0, 2, 1).unsqueeze(-1)  # (N, seq_len, num_nodes, 1)
        X_val = X_val.permute(0, 2, 1).unsqueeze(-1)
        X_test = X_test.permute(0, 2, 1).unsqueeze(-1)
        num_nodes = X_train.shape[2]
        input_dim = X_train.shape[3]
    elif args.method == 'STGCN':

        X_train, X_val, X_test = X_train.transpose(0,2,1), X_val.transpose(0,2,1), X_test.transpose(0,2,1)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
    
        X_train = X_train.unsqueeze(-1)  # (N, num_nodes, seq_len, 1)
        X_val = X_val.unsqueeze(-1)
        X_test = X_test.unsqueeze(-1)
        # import pdb; pdb.set_trace()

        '''
        reshaped_data = X_train.reshape(-1, X_train.shape[1]).cpu().numpy()
        correlation_matrix = np.corrcoef(reshaped_data, rowvar=False)
        
        threshold = 0.5
        adjacency_matrix = np.where(np.abs(correlation_matrix) >= threshold, 1, 0)
        np.fill_diagonal(adjacency_matrix, 1)
        print("adjacency matrix",adjacency_matrix.shape)
        np.save('adjacency_matrix_STGCN.npy',adjacency_matrix)
        '''
        if args.adj_type == 'train':
            adjacency_matrix = get_adj(X_train)
        elif args.adj_type == 'all':
            X_all =torch.cat([X_train,X_val,X_test],dim=0)
            adjacency_matrix = get_adj(X_all)
        elif args.adj_type == 'seperate':
            adjacency_matrix = get_adj(X_train)
            adj_val = get_adj(X_val)
            A_hat_val = normalize_adj(adj_val)
            adj_test = get_adj(X_test)
            A_hat_test = normalize_adj(adj_test)
            A_hat_val = torch.from_numpy(A_hat_val).float().to(device)
            A_hat_test = torch.from_numpy(A_hat_test).float().to(device)

        
        A_hat = normalize_adj(adjacency_matrix)
     
        A_hat = torch.from_numpy(A_hat).float().to(device)
        
        num_nodes = X_train.shape[1]
        num_features = X_train.shape[3]
        num_timesteps_input = X_train.shape[2]
        
    else:
        # keep the shape
        X_train = X_train.transpose(0, 2, 1)  # (N, n_channels, seq_len)
        X_val = X_val.transpose(0, 2, 1)
        X_test = X_test.transpose(0, 2, 1)
        nchannel = X_train.shape[1]
        seq_len = X_train.shape[2]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    if args.method == 'MTGNN':
        X_train, X_val, X_test = X_train.unsqueeze(1), X_val.unsqueeze(1), X_test.unsqueeze(1)
    
    print(X_train.shape)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # import pdb; pdb.set_trace()

    
    if args.method == 'DTW':
       
        pass
    else:
        if args.method == 'FCN':
            model = FCNModel(nchannel, n_classes).to(device)
        elif args.method == 'CNN':
            # model = CNNModel(X_train.shape[2], nchannel).to(device)
            model = CNNModel(nchannel, seq_len, n_classes).to(device)
        elif args.method == 'LSTM':
            model = LSTMModel(nchannel, seq_len, n_classes).to(device)
        elif args.method == 'CNNLSTM':
            model = CNNLSTM(nchannel, seq_len, n_classes).to(device)
        elif args.method == 'DeepConvNet':
            model = DeepConvNet(nchannel, seq_len, n_classes).to(device)
        elif args.method == 'ResNet1D':
            model = ResNet1D(nchannel, n_classes).to(device)
        elif args.method == 'SimpleEEGNet':
            model = SimpleEEGNet(nchannel, n_classes).to(device)
        elif args.method == 'LVM':
            model = LVMModel(in_channels=nchannel, seq_len=seq_len, num_classes=n_classes).to(device)
        elif args.method == 'BiGRU':
            model = BiGRUModel(in_channels=nchannel, seq_len=seq_len, num_classes=n_classes).to(device)
        elif args.method == 'XGBoost':
            model = XGBoostModel(num_classes=n_classes).to(device)
        elif args.method == 'STGCN':
            model = STGCN(
                num_nodes=num_nodes,
                num_features=num_features,
                num_timesteps_input=num_timesteps_input,
                num_classes=n_classes
            ).to(device)
           

        elif args.method == 'AGCRN':
            
            num_nodes = X_train.shape[2]
            input_dim = X_train.shape[3]
            rnn_units = 32   
            num_layers = 2   
            embed_dim = 10   
            cheb_k = 2       
            num_classes = n_classes  
            model = AGCRN(
                num_nodes=num_nodes,
                input_dim=input_dim,
                rnn_units=rnn_units,
                num_layers=num_layers,
                embed_dim=embed_dim,
                cheb_k=cheb_k,
                num_classes=num_classes
            ).to(device)

        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        # # calculate model parameters
        # total_params = sum(p.numel() for p in model.parameters())
        # # turn to MB
        # total_params = total_params / 1024 / 1024
        # logger.debug(f"Model parameters: {total_params:.2f} MB")
        # # exit()
        # exit()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_val_acc = 0.0
        best_val_sensitivity = 0.0
        best_val_f1 = 0.0
        best_test_acc = 0.0
        best_test_f1 = 0.0
        best_test_precision = 0.0
        best_test_recall = 0.0
        best_test_sensitivity = 0.0
        best_test_specificity = 0.0
        best_test_auc = 0.0
        best_test_epoch = 0
        uid = uuid4()
        name = args.method + '_' + args.dataset + '_' + str(args.smooth) + '_' + str(args.denoise) + '_' + args.adj_type + '_' + str(args.learning_rate) + '_' + str(args.num_epochs)
        best_model_dir = './checkpoints/'
        best_model_path = os.path.join(best_model_dir, name)
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            n_loader = 0
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                # import pdb; pdb.set_trace()
                if args.method == 'STGCN':
                    outputs = model(A_hat, inputs)
                   
                else:
                    outputs = model(inputs)
                    #print(outputs.shape)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                n_loader += 1
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            if args.method == 'STGCN' and args.adj_type == 'seperate':
                val_acc, val_f1, val_precision, val_recall, val_sensitivity, val_specificity, val_auc = evaluate(model, val_loader, device, A_hat_val)
                test_acc, test_f1, test_precision, test_recall, test_sensitivity, test_specificity, test_auc = evaluate(model, test_loader, device, A_hat_test)
            elif args.method == 'STGCN' and args.adj_type != 'seperate':
                val_acc, val_f1, val_precision, val_recall, val_sensitivity, val_specificity, val_auc = evaluate(model, val_loader, device, A_hat)
                test_acc, test_f1, test_precision, test_recall, test_sensitivity, test_specificity, test_auc = evaluate(model, test_loader, device, A_hat)
            else:
                val_acc, val_f1, val_precision, val_recall, val_sensitivity, val_specificity, val_auc = evaluate(model, val_loader, device)
                test_acc, test_f1, test_precision, test_recall, test_sensitivity, test_specificity, test_auc = evaluate(model, test_loader, device)


            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)  
            
            # Track best test metrics
            if test_f1 > best_test_f1:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                best_test_precision = test_precision
                best_test_recall = test_recall
                best_test_sensitivity = test_sensitivity
                best_test_specificity = test_specificity
                best_test_auc = test_auc
                best_test_epoch = epoch + 1

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            logger.debug(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
            logger.debug(f'Validation -> Accuracy: {val_acc:.4f}, F1-score: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}, AUC: {val_auc:.4f}')
            logger.debug(f'Test -> Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, Sensitivity: {test_sensitivity:.4f}, Specificity: {test_specificity:.4f}, AUC: {test_auc:.4f}')

        model.load_state_dict(torch.load(best_model_path))
        
        if args.method == 'STGCN' and args.adj_type == 'seperate':
            test_acc, test_f1, test_precision, test_recall, test_sensitivity, test_specificity, test_auc = evaluate(model, test_loader, device, A_hat_test)
        elif args.method == 'STGCN' and args.adj_type != 'seperate':
            test_acc, test_f1, test_precision, test_recall, test_sensitivity, test_specificity, test_auc = evaluate(model, test_loader, device, A_hat)
        else:
            test_acc, test_f1, test_precision, test_recall, test_sensitivity, test_specificity, test_auc = evaluate(model, test_loader, device)


        logger.debug(f'Test Results -> Accuracy: {test_acc:.4f}, F1-score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, Sensitivity: {test_sensitivity:.4f}, Specificity: {test_specificity:.4f}, AUC: {test_auc:.4f}')

        # Write results to the results file
        with open(results_file, 'a') as f:
            # Write hyperparameters
            f.write(f"Hyperparameters: dataset={args.dataset}, method={args.method}, seed={args.seed}, "
                   f"batch_size={args.batch_size}, learning_rate={args.learning_rate}, "
                   f"num_epochs={args.num_epochs}, adj_type={args.adj_type}\n")
            # Write metrics and best epoch
            f.write(f"Metrics: best_epoch={best_epoch}, accuracy={test_acc:.4f}, f1={test_f1:.4f}, "
                   f"precision={test_precision:.4f}, recall={test_recall:.4f}, "
                   f"sensitivity={test_sensitivity:.4f}, specificity={test_specificity:.4f}, auc={test_auc:.4f}\n")
            # Write best test metrics during training
            f.write(f"Best Test Metrics: epoch={best_test_epoch}, accuracy={best_test_acc:.4f}, f1={best_test_f1:.4f}, "
                   f"precision={best_test_precision:.4f}, recall={best_test_recall:.4f}, "
                   f"sensitivity={best_test_sensitivity:.4f}, specificity={best_test_specificity:.4f}, auc={best_test_auc:.4f}\n\n")
