from abc import ABC
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, GINConv, SAGEConv, GATConv, AvgPooling
from torch import optim
from tqdm import tqdm
import time
import random
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import shutil
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar


# 定义消息传递函数，将节点特征 'h' 与边特征 'edge_weight' 相乘，并存储在新的边特征 'm' 中
msg_func = fn.u_mul_e('h', 'edge_weight', 'm') 
reduce_mean = fn.mean('m', 'h') #计算邻居节点的 'm' 特征的平均值，并将结果存储在节点的 'h' 特征中
reduce_sum = fn.sum('m', 'h') #计算邻居节点的 'm' 特征的总和，并将结果存储在节点的 'h' 特征中
reduce_max = fn.max('m', 'h') #计算邻居节点的 'm' 特征的最大值，并将结果存储在节点的 'h' 特征中


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-4] + "_best.pth")


# with dropout and bn
class MLP(nn.Module, ABC):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1, bn=False):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bn = bn
        self.fc = torch.nn.ModuleList()

        if self.num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif self.num_layers == 1:
            self.fc.append(nn.Linear(input_dim, output_dim))
        else:
            self.fc.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 1):
                self.fc.append(nn.Dropout(p=dropout))
                self.fc.append(nn.ReLU())
                if self.bn:
                    self.fc.append(nn.BatchNorm1d(self.hidden_dim))
                if layer < num_layers - 2:
                    self.fc.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        return self.fc(x)


def create_file_logger(file_name: str = 'log.txt', log_format: str = '%(message)s', log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    handler = logging.FileHandler(file_name, "w")
    handler.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(log_level)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


class LoggerNone:
    def __init__(self) -> None:
        pass
    
    def info(self, input_, *args, **kwargs):
        pass


class NodeApplyModule(nn.Module):
    # Update node feature h_v with (Wh_v+b)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}
    

class GNNLayer(nn.Module):
    """
        Param: [in_dim, out_dim]
    """

    def __init__(self, in_dim, out_dim, activation=nn.ReLU(), dropout=0.3, 
                 batch_norm=False, residual=False, dgl_builtin=True, gnn_type="gcn"):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.dgl_builtin = dgl_builtin
        self.gnn_type = gnn_type

        if in_dim != out_dim:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if self.dgl_builtin == False:
            self.apply_mod = NodeApplyModule(in_dim, out_dim)
        else:
            print(gnn_type)
            if gnn_type == "gcn":
                self.conv = GraphConv(in_dim, out_dim)
            elif gnn_type == "gin":
                self.conv = GINConv(apply_func=nn.Linear(in_dim, out_dim), aggregator_type="sum")
            elif gnn_type == "sage":
                self.conv = SAGEConv(in_dim, out_dim, aggregator_type="mean")
            elif gnn_type == "gat":
                self.conv = GATConv(in_dim, out_dim, 1, 0.1, 0.1)
            else:
                exit()

    def forward(self, g, feature, edge_weight=None):
        h_in = feature  # to be used for residual connection

        if self.dgl_builtin == False:
            g.ndata['h'] = feature
            # g.update_all(msg, reduce)
            g.update_all(message_func=msg_func, reduce_func=reduce_mean)
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata['h']  # result of graph convolution
        else:
            # g = dgl.add_self_loop(g)
            h = self.conv(g, feature, edge_weight=edge_weight)
        if self.gnn_type == "gat":
            h = h.squeeze(1)

        if self.batch_norm:
            h = self.batchnorm_h(h)  # batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        h = self.dropout(h)

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channels,
                                                                         self.out_channels, self.residual)


class GNNNet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args_ = args
        in_dim = args.rna_in_dim
        hidden_dim = args.hidden_dim
        out_dim = args.out_dim
        batch_norm = args.batch_norm
        dropout = args.dropout
        self.n_layers = args.gnn_n_layers
        rbp_feat_model = args.rbp_feat_model
        dim_map = {"esm1_t12_85M_UR50S": 768, "esm2_t33_650M_UR50D": 1280, "esm2_t36_3B_UR50D": 2560}
        rbp_path = f'/data0/gaoyifei/GraphProt_byRBP/a_RBP_feats/{rbp_feat_model}/rbp_features_standard.pt'
        rbp_in_dim = dim_map[rbp_feat_model]
        self.precomputed_rbp_features = torch.load(rbp_path)
        self.temperature = args.temperature
        self.loss_type = args.loss_type
    
        # GNN start
        self.mlp_rna = MLP(in_dim, hidden_dim, out_dim, 2, 0.1, False)
        self.mlp_protein = MLP(rbp_in_dim, hidden_dim, out_dim, 2, 0.1, False)
        
        self.layers_gnn = nn.ModuleList()
        for _ in range(self.n_layers): 
            self.layers_gnn.append(GNNLayer(out_dim, out_dim, F.leaky_relu, dropout, batch_norm, gnn_type=args.gnn))
        self.readout = AvgPooling()

        """use protein or not"""
        if not args.use_binary_protein:
            self.mlp_binary = nn.Linear(out_dim, 2)
        else:
            self.mlp_binary = MLP(out_dim+out_dim, out_dim, 2, 2, 0.1, False)
        
    def forward(self, g, h, edge_weight=None, protein_h=None):  # g:batch_graphs
        # naive GNN, no edge feat
        h_rna = h  # g.ndata['h'], should be the same
        # batch_size = len(g.batch_num_nodes())  #128
        h_rna = self.mlp_rna(h_rna)
        for i in range(self.n_layers):
            h1 = self.layers_gnn[i](g, h_rna, edge_weight=edge_weight)
        # g.ndata['h'] = h1  # update
        h_final = self.readout(g, h1)

        if protein_h is not None:
            protein_h = self.mlp_protein(protein_h)
        return h_final, protein_h

    """rna binding 0/1"""
    def binary_loss(self, g, h, edge_weight, label1, label2_, protein_h_id):
        if not self.args_.use_binary_protein:
            pred, _ = self.forward(g, h, edge_weight)
        else:
            protein_h = self.precomputed_rbp_features[label2_].to(h.device)
            rna_h, protein_h = self.forward(g, h, edge_weight, protein_h)
            pred = torch.cat([rna_h, protein_h], dim=1)
        pred = self.mlp_binary(pred)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label1) #pred:(batch_size,2) label:(batch_size)
        return loss, pred
    
    """tensor input"""
    def cosine_similarity(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        
        # Compute cosine similarity
        cosine_sim = torch.mm(x, y.T)
        return cosine_sim
    
    def cosine_loss(self, A, B, L, margin=0.001):
        
        cosine_sim = self.cosine_similarity(A, B)
        Lpos = torch.clamp(L, min=0)
        Lneg = torch.clamp(L, max=0)
        pos_loss = 19*(1 - cosine_sim) * Lpos
        neg_loss = (torch.clamp(cosine_sim - margin, min=0)) * (-Lneg)
        total_loss = pos_loss+neg_loss
        # 计算每一行的和
        row_sums = total_loss.sum(dim=1)

        final_loss = torch.mean(row_sums / 19)
        return final_loss, cosine_sim

    def infonce_loss(self, A, B, L, train_nums=None, temperature=1.0, down_add_up=True):
        L = L.clamp(min=0).T #把L转成0/1
        
        C = self.cosine_similarity(B, A)/temperature #[batch_size,dataset_num]
        """计算每个类别的权重"""
        temp_ = 1. / (L.sum(dim=1)+1)
        weights = temp_ / torch.sum(temp_)
        
        mask_neg = 1-L if not down_add_up else torch.ones(L.shape, device=L.device) #[batch_size,dataset_num]
        
        # mask similarity, sum of log, mask from protein, apply weights from the view of proteins
        t_ = torch.multiply(torch.exp(C), L)

        final_loss_ = -torch.sum(torch.multiply(torch.log(torch.sum(t_, dim=-1) / (torch.sum(torch.multiply(torch.exp(C), mask_neg), dim=-1) + 1e-6) + 1e-6), L.any(1)*weights))
        """/len(A) or len(B)"""
        return final_loss_ / len(B), C

    def rbp_loss(self, g, h, edge_weight, labels2, protein_h_id, use_protein_weight=False):
        all_train_nums = torch.as_tensor([
            92346,
            2410,
            16041,
            322034,
            31012,
            66061,
            41266,
            26780,
            3709,
            88274,
            17343,
            19418,
            13904,
            18508,
            36633,
            167110,
            34184,
            78984,
            40980
        ])
        train_nums = all_train_nums[protein_h_id] if use_protein_weight else None
        protein_h = self.precomputed_rbp_features[protein_h_id].to(h.device)
        rna_h, protein_h = self.forward(g, h, edge_weight, protein_h)
        if self.loss_type == 'cosine':
            final_loss, c_ = self.cosine_loss(rna_h, protein_h, labels2)  # .cpu().detach()
        elif self.loss_type == 'infonce':
            final_loss, c_ = self.infonce_loss(rna_h, protein_h, labels2, train_nums=train_nums, temperature=self.temperature)

        return final_loss, c_

    def find_most_similar_rbp(self, g, edge_weight, protein_id, label2=None):
        id_mapping = {}
        for id_ in range(len(protein_id)):
            id_mapping[id_] = protein_id[id_].item()

        protein_h = self.precomputed_rbp_features[protein_id].to(g.device)
        with torch.no_grad():
            rna_h, protein_h = self.forward(g, g.ndata["h"], edge_weight, protein_h)
            if not self.args_.use_binary_protein:
                pred = rna_h
            else:
                protein_h_all = self.precomputed_rbp_features[label2].to(rna_h.device)
                protein_h_all = self.mlp_protein(protein_h_all)
                pred = torch.cat([rna_h, protein_h_all], dim=1)
            binary_pred = self.mlp_binary(pred)
        cos_s = self.cosine_similarity(rna_h, protein_h)
        res_values, res_indices = torch.sort(cos_s, dim=1, descending=True)
        for id_ in range(len(protein_id)-1, -1, -1):
            res_indices = torch.where(res_indices==id_, torch.as_tensor(id_mapping[id_]), res_indices)
        m = nn.Softmax(dim=1)
        return cos_s, res_values, res_indices, m(binary_pred), rna_h.cpu(), protein_h.cpu()

    def get_all_rbp(self, g, edge_weight, protein_id, label2=None):
        id_mapping = {}
        for id_ in range(len(protein_id)):
            id_mapping[id_] = protein_id[id_].item()

        protein_h = self.precomputed_rbp_features[protein_id].to(g.device)
        with torch.no_grad():
            rna_h, protein_h = self.forward(g, g.ndata["h"], edge_weight, protein_h)
            cos_s = self.cosine_similarity(rna_h, protein_h)
            if not self.args_.use_binary_protein:
                pred = rna_h
            else:
                protein_h_all = self.precomputed_rbp_features[protein_id].to(rna_h.device)
                protein_h_all = self.mlp_protein(protein_h_all)
                protein_h_all = protein_h_all.expand((len(rna_h), -1, -1))
                protein_h_all = torch.permute(protein_h_all, (1, 0, 2))
                rna_h = rna_h.expand((len(protein_id), -1, -1))
                pred = torch.cat([rna_h, protein_h_all], dim=-1)
            binary_pred = self.mlp_binary(pred)
        m = nn.Softmax(dim=-1)
        # return cos_s, res_values, res_indices, m(binary_pred)
        return cos_s, None, None, m(binary_pred)

    def get_embedding_(self, g, edge_weight, protein_id):
        id_mapping = {}
        for id_ in range(len(protein_id)):
            id_mapping[id_] = protein_id[id_].item()

        protein_h = self.precomputed_rbp_features[protein_id].to(g.device)
        with torch.no_grad():
            rna_h, protein_h = self.forward(g, g.ndata["h"], edge_weight, protein_h)
        return rna_h, protein_h

class Trainer:
    def __init__(self, args, model=None) -> None:
        self.args = args
        self.model = model.to(self.args.device)
        self.dir_prefix = f"{args.protein_id}_{args.batch_size}_{args.lr}_{args.use_lp_weight}_{args.lp_eps}_{args.epochs}_{args.gnn}_{args.use_cl_loss}_{args.use_binary_loss}_{args.use_binary_protein}"
        log_path = os.path.join("/home/gaoyifei/RNARepresentation/out/GCN", "logs_leave_one", self.dir_prefix+".log")
        writer_path = os.path.join("/home/gaoyifei/RNARepresentation/out/GCN", "runs_leave_one", self.dir_prefix)
        if args.gnn != "gcn":
            self.dir_prefix = f"{args.gnn}_{args.batch_size}_{args.lr}_{args.use_lp_weight}_{args.lp_eps}_{args.epochs}_{args.gnn}_{args.use_cl_loss}_{args.use_binary_loss}_{args.use_binary_protein}"
            log_path = os.path.join(f"/home/gaoyifei/RNARepresentation/out/{args.gnn}", "10_10", self.dir_prefix+".log")
            writer_path = os.path.join(f"/home/gaoyifei/RNARepresentation/out/{args.gnn}", "10_10", self.dir_prefix)
        if not os.path.exists(writer_path):
            os.makedirs(writer_path)
        self.logger = create_file_logger(log_path) if args.save else LoggerNone()
        if args.save == 1:
            # mkdir if needed
            self.writer = SummaryWriter(writer_path)
        else:
            self.writer = None
        self.logger.info(f"======={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")
        self.logger.info("=======Setting=======")
        for k in args.__dict__:
            v = args.__dict__[k]
            self.logger.info("{}: {}".format(k, v))
        self.logger.info("=======Training=======")
        self._set_optimizer()

    def _set_optimizer(self):
        self.model_param_group = [{'params': self.model.parameters(),
                                   'lr': self.args.lr}]
        if self.args.opt == "adam":
            self.optimizer = optim.Adam(self.model_param_group, weight_decay=self.args.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model_param_group, lr=self.args.lr,
                                       weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        if self.args.lr_type == "reduce":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                  factor=self.args.lr_reduce_factor,
                                                                  patience=self.args.lr_reduce_patience,
                                                                  verbose=True)
        elif self.args.lr_type == "step":
            self.scheduler = optim.lr_scheduler. \
                MultiStepLR(self.optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        else:
            self.scheduler = optim.lr_scheduler. \
                CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=self.args.lr * self.args.cos_rate)

    def training(self, data_loader, epoch, total_step, protein_id, tag, eval_every=10):
        assert tag in ("training", "validation", "test")
        assert (self.args.use_binary_loss or self.args.use_cl_loss)
        start_t = time.time()
        if tag == "training":
            self.model.train()
        else:
            self.model.eval()
        loss_accumulation = 0.
        rea_num = 0
        # total, binary, cl
        tmp_loss = torch.zeros((3,))
        conf_matrix_1, conf_matrix_2 = np.zeros((2, 2)), np.zeros((2, 2))
        
        for step, (batch_g, label1, label2) in enumerate(tqdm(data_loader)):
            time_0 = time.time()
            temp_num = batch_g.batch_size
            rea_num += temp_num
            label2_ = torch.argmax((label2!=-1).float(), dim=1)
            batch_g, label1, label2 = batch_g.to(self.args.device), label1.to(self.args.device), label2.to(self.args.device)
            
            if tag == "training":
                edge_weight_ = batch_g.edata["edge_weight"] if self.args.use_lp_weight else None
                # res_h, _ = model(batch_dgl, batch_dgl.ndata["h"], edge_weight=edge_weight_)
                loss_binary, pred_ = self.model.binary_loss(batch_g, batch_g.ndata["h"], edge_weight_, label1, label2_, protein_h_id=protein_id) if self.args.use_binary_loss else torch.as_tensor(0.)
                if self.args.use_cl_loss:
                    loss_cl, c_ = self.model.rbp_loss(batch_g, batch_g.ndata["h"], edge_weight_, label2, protein_h_id=protein_id, use_protein_weight=True)
                else:
                    loss_cl = torch.as_tensor(0.)
            else:
                with torch.no_grad():
                    edge_weight_ = batch_g.edata["edge_weight"] if self.args.use_lp_weight else None
                    # res_h, _ = model(batch_dgl, batch_dgl.ndata["h"], edge_weight=edge_weight_)
                    loss_binary, pred_ = self.model.binary_loss(batch_g, batch_g.ndata["h"], edge_weight_, label1, label2_, protein_h_id=protein_id) if self.args.use_binary_loss else torch.as_tensor(0.)
                    if self.args.use_cl_loss:
                        loss_cl, c_ = self.model.rbp_loss(batch_g, batch_g.ndata["h"], edge_weight_, label2, protein_h_id=protein_id, use_protein_weight=True)
                    else:
                        loss_cl = torch.as_tensor(0.)
            loss = loss_binary + loss_cl
            if torch.isnan(loss):
                print("is nan")
            # backprop
            if tag == "training":
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
            pred_ = torch.argmax(pred_, dim=1)
            
            # cos_s, res_values, res_indices = self.model.find_most_similar_rbp(batch_g, edge_weight_, protein_id)
            """conf matrix"""
            conf_matrix_1 += confusion_matrix(label1.cpu().detach().numpy(), pred_.cpu().detach().numpy(), labels=[0, 1])
            if self.args.use_cl_loss:
                conf_matrix_2 += confusion_matrix(torch.flatten(torch.clamp(label2, 0.)).cpu().detach().numpy(), torch.flatten(c_.T>self.args.cutoff_cos).cpu().detach().numpy(), labels=[0, 1])
            # pred_ == label1
            # torch.flatten(cos_s>0.3) == torch.flatten(torch.clamp(label2, 0.))
            loss_item, binary_item, cl_item = loss.item(), loss_binary.item(), loss_cl.item()
            tmp_loss[0] += loss_item
            tmp_loss[1] += binary_item
            tmp_loss[2] += cl_item
            loss_accumulation += (loss_item * temp_num)
            if (step + 1) % eval_every == 0:
                self.logger.info('Epoch: {:03d} Step: {:06d} total: {:.6f} binary: {:.6f} cl: {:.6f} time: {:.2f}s\nconf_1: {} conf_2: {}'
                            .format(epoch, step + 1, tmp_loss[0] / eval_every, tmp_loss[1] / eval_every,
                                    tmp_loss[2] / eval_every, time.time() - start_t, conf_matrix_1.tolist(), conf_matrix_2.tolist()))
                tmp_loss = torch.zeros((3,))
            if self.writer is not None:
                self.writer.add_scalar("loss/{}".format(tag), loss_item, total_step + step + 1)
                self.writer.add_scalar("loss/{}_binary".format(tag), binary_item, total_step + step + 1)
                self.writer.add_scalar("loss/{}_cl".format(tag), cl_item, total_step + step + 1)
            
        return loss_accumulation / rea_num, time.time() - start_t, total_step + step + 1, conf_matrix_1.tolist(), conf_matrix_2.tolist()

    def get_similarity(self, data_loader, protein_id):
        assert (self.args.use_binary_loss or self.args.use_cl_loss)
        self.model.eval()
        id_mapping = {}
        for id_ in range(len(protein_id)):
            id_mapping[id_] = protein_id[id_].item()
        """rna accuracy, and protein accuracy"""
        pos_cos, neg_cos = [], []
        protein_cos, protein_label = [], []
        binary_label, binary_pred = [], []
        rna_hs = []
        for step, (batch_g, label1, label2) in enumerate(tqdm(data_loader)):
            label2_all = torch.argmax((label2!=-1).float(), dim=1)
            batch_g, label2 = batch_g.to(self.args.device), label2.to(self.args.device)
            with torch.no_grad():
                edge_weight_ = batch_g.edata["edge_weight"] if self.args.use_lp_weight else None
                cos_s, res_values, res_indices, pred_prob, rna_h, protein_h = self.model.find_most_similar_rbp(batch_g, edge_weight_, protein_id, label2_all)
            cos_s, label2, pred_prob = cos_s.cpu(), label2.cpu(), pred_prob.cpu()
            binary_label.append(label1)
            binary_pred.append(pred_prob)
            protein_cos.append(cos_s)
            rna_hs.append(rna_h)
            cos_s_ =  torch.flatten(cos_s)
            label2_ = torch.clamp(label2, -2.)  # 0. or not
            protein_label.append(label2_)
            label2_ = torch.flatten(label2_)
            pos_cos.append(cos_s_[label2_==1.])
            neg_cos.append(cos_s_[label2_==0.])
        protein_cos = torch.cat(protein_cos, dim=0)
        protein_label = torch.cat(protein_label, dim=0)
        pos_distribution = torch.cat(pos_cos, dim=0).numpy()
        neg_distribution = torch.cat(neg_cos, dim=0).numpy()
        if neg_distribution.shape[0] != 0:
            pos_kde = gaussian_kde(pos_distribution)
            neg_kde = gaussian_kde(neg_distribution)
            def find_intersection(x):
                return abs(pos_kde(x) - neg_kde(x))
            x_min = min(pos_distribution.min(), neg_distribution.min())
            x_max = max(pos_distribution.max(), neg_distribution.max())
            result = minimize_scalar(find_intersection, bounds=(x_min, x_max), method='bounded')
            if find_intersection(result.x) < 1e-3:
                threshold = result.x
            else:
                threshold = [(pos_distribution.mean() + neg_distribution.mean()) / 2]
        else:
            threshold = 0.
        return pos_distribution, neg_distribution, threshold, protein_cos, protein_label, id_mapping, torch.cat(binary_label, dim=0).numpy(), torch.cat(binary_pred, dim=0).numpy(), torch.cat(rna_hs, dim=0).numpy(), protein_h.numpy()

    def get_rna2allproteins(self, data_loader, protein_id):
        assert (self.args.use_binary_loss or self.args.use_cl_loss)
        self.model.eval()
        id_mapping = {}
        for id_ in range(len(protein_id)):
            id_mapping[id_] = protein_id[id_].item()
        """rna accuracy, and protein accuracy"""
        pos_cos, neg_cos = [], []
        protein_cos, protein_label = [], []
        binary_label, binary_pred = [], []
        for step, (batch_g, label1, label2) in enumerate(tqdm(data_loader)):
            label2_all = torch.argmax((label2!=-1).float(), dim=1)
            batch_g, label2 = batch_g.to(self.args.device), label2.to(self.args.device)
            with torch.no_grad():
                edge_weight_ = batch_g.edata["edge_weight"] if self.args.use_lp_weight else None
                cos_s, res_values, res_indices, pred_prob = self.model.get_all_rbp(batch_g, edge_weight_, protein_id, label2_all)
            cos_s, label2, pred_prob = cos_s.cpu(), label2.cpu(), pred_prob.cpu()
            return cos_s, label2.squeeze(), pred_prob
        
    def get_embedding(self, data_loader, protein_id):
        assert (self.args.use_binary_loss or self.args.use_cl_loss)
        self.model.eval()
        id_mapping = {}
        for id_ in range(len(protein_id)):
            id_mapping[id_] = protein_id[id_].item()
        pos_cos, neg_cos = [], []
        protein_cos, protein_label = [], []
        binary_label, binary_pred = [], []
        for step, (batch_g, label1, label2) in enumerate(tqdm(data_loader)):
            batch_g, label2 = batch_g.to(self.args.device), label2.to(self.args.device)
            with torch.no_grad():
                edge_weight_ = batch_g.edata["edge_weight"] if self.args.use_lp_weight else None
                rna_h, protein_h = self.model.get_embedding_(batch_g, edge_weight_, protein_id)
            cos_s, label2, pred_prob = cos_s.cpu(), label2.cpu(), pred_prob.cpu()
            binary_label.append(label1)
            binary_pred.append(pred_prob)
            protein_cos.append(cos_s)
            cos_s_ =  torch.flatten(cos_s)
            label2_ = torch.clamp(label2, -2.)  # 0. or not
            protein_label.append(label2_)
            label2_ = torch.flatten(label2_)
            pos_cos.append(cos_s_[label2_==1.])
            neg_cos.append(cos_s_[label2_==0.])
        protein_cos = torch.cat(protein_cos, dim=0)
        protein_label = torch.cat(protein_label, dim=0)
        return protein_cos, protein_label, id_mapping, torch.cat(binary_label, dim=0).numpy(), torch.cat(binary_pred, dim=0).numpy()

    def save(self, epoch, best_p, is_best, model_save_dir, tag=""):
        save_dict = {"epoch": epoch, "best_performance": best_p, "optimizer": self.optimizer.state_dict(),
                     "model_state_dict": self.model.state_dict()}
        save_checkpoint(
            save_dict, is_best,
            os.path.join(model_save_dir, "models", self.dir_prefix+tag+".pth"))
    