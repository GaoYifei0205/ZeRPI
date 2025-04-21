import os
import pickle
import numpy as np
from utils import *
from model import *
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef


if __name__ == '__main__':
    data_prefix = "/data0/gaoyifei/GraphProt_byRBP"
    id2rbp, rbp2id = {}, {}
    for i in range(len(rbps)):
        id2rbp[i] = rbps[i]
        rbp2id[rbps[i]] = i

    args_ = get_args()
    args_.save = 0
    args_.use_binary_protein = True
    # args_.use_lp_weight = False
    # args_.lp_eps = 0.1
    args_.epochs = 50
    args_.device = "cuda:0"
    args_.rbp_feat_model = "esm2_t33_650M_UR50D"
    seed_torch(args_.seed)
    dir_prefix = f"{args_.batch_size}_{args_.lr}_{args_.use_lp_weight}_{args_.lp_eps}_{args_.epochs}_{args_.gnn}_{args_.use_cl_loss}_{args_.use_binary_loss}_{args_.use_binary_protein}"
    log_path = os.path.join("/home/gaoyifei/RNARepresentation/out/GCN/zero_shot", dir_prefix+".txt")
    logger = create_file_logger(log_path)

    zero_shot_rbps = ['AGO2', 'ELAV1', 'ALKB5', 'FUS', 'NCBP3', 'PTBP1', 'RBP56', 'TADBP', 'TIAR']
    true_id = [rbps.index(one) for one in zero_shot_rbps]
    protein_id = torch.as_tensor([rbp2id[one] for one in zero_shot_rbps])    
    zero_shot_dataset = []
    p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", "others", f"{args_.use_lp_weight}_{args_.lp_eps}")
    zero_shot_dataloader = []
    for rbp_i in true_id:
        temp_p = os.path.join(p_, f"test_batch_{rbps[rbp_i]}.bin")
        batch_g, labels_dict = load_graphs(temp_p)
        zero_shot_dataloader.append([dgl.batch(batch_g), labels_dict["label1"], labels_dict["label2"][:, protein_id]])
    
    args_.gnn = "gat"
    # args_.gnn = "gin"
    args_.device = 4
    model = GNNNet(args=args_)
    model_prefix = "/data0/gaoyifei/leave_one_res/GCN/models_old_10_9/models"
    p_ = os.path.join(model_prefix, f"{dir_prefix}_best.pth")
    p_ = "/data0/gaoyifei/10/models/gat_512_0.001_False_0.1_50_gat_True_True_True_best.pth"
    # p_ = "/data0/gaoyifei/10/GCN/models/gin_512_0.001_False_0.1_30_gin_True_True_True_best.pth"
    if not os.path.exists(p_):
        print(f"The model path '{p_}' does not exist.")
        exit()
    model.load_state_dict(torch.load(p_)["model_state_dict"])
    logger.info(f"loading pretrained weights from {p_}")
    trainer_ = Trainer(args=args_, model=model)

    th_val = [0.]
    pos_test, neg_test, th_test, protein_cos_test, protein_label_test, id_mapping_test, binary_label_test, binary_pred_test_prob, rna_feats, protein_feats = trainer_.get_similarity(zero_shot_dataloader, protein_id)
    

    # split; merge is bad
    rbp_name_map = {'ALKB5':'ALKBH5', 'NCBP3':'C17ORF85','AGO2':'AGO2', 'ELAV1':'ELAVL1', 'TADBP':'TDP43', 'TIAR':'TIAL1', 
                    'FUS':'FUS', 'RBP56': 'TAF15', 'PTBP1': 'PTBv1','CAPR1': 'CAPRIN1', 'EWS': 'EWSR1', 'HNRPC': 'HNRNPC', 
                    'MOV10': 'MOV10', 'PUM2': 'PUM2', 'QKI': 'QKI', 'RTCB': 'C22ORF28', 'SRSF1': 'SFRS1', 'TIA1': 'TIA1', 
                    'Z3H7B': 'ZC3H7B'}
    from collections import defaultdict
    target_all = defaultdict(list)
    rna_feats_all = defaultdict(list)
    for i, one in enumerate(protein_label_test):
        if 1 in one:
            id_ = int(torch.where(one == 1)[0][0])
            target_all[id_].append(1)
            rna_feats_all[id_].append(rna_feats[i])
        elif 0 in one:
            id_ = int(torch.where(one == 0)[0][0])
            target_all[id_].append(0)
            rna_feats_all[id_].append(rna_feats[i])
        else:
            raise Exception
    for id_ in range(9):
        target_all[id_] = np.array(target_all[id_]+[2])
    
    binary_auc = roc_auc_score(binary_label_test, binary_pred_test_prob[:, 1])
    binary_pr = average_precision_score(binary_label_test, binary_pred_test_prob[:, 1])
    binary_pred_test = np.argmax(binary_pred_test_prob, axis=1)
    conf_matrix_binary = confusion_matrix(binary_label_test, binary_pred_test, labels=[0, 1])
    binary_ = classification_report(binary_label_test, binary_pred_test, target_names=["binary 0", "binary 1"], digits=4)
    logger.info(f"val threshold: {th_val[0]}")
    logger.info("binary:")
    logger.info(conf_matrix_binary)
    logger.info(binary_)
    logger.info(f"roc_auc: {binary_auc}")
    logger.info(f"apr: {binary_pr}")

    rna_label = np.concatenate([np.ones(pos_test.shape), np.zeros(neg_test.shape)], axis=0)
    rna_temp = np.concatenate([pos_test, neg_test], axis=0)
    rna_pred = (rna_temp>th_val[0])
    rna_auc = roc_auc_score(rna_label, (rna_temp+1)/2)
    conf_matrix_rna = confusion_matrix(rna_label, rna_pred, labels=[0, 1])
    rna_ = classification_report(rna_label, rna_pred, target_names=["rna 0", "rna 1"], digits=4)

    logger.info("\nall:")
    logger.info(conf_matrix_rna)
    logger.info(rna_)
    logger.info(f"roc_auc: {rna_auc}")

    acc_ = accuracy_score(rna_label, rna_pred)
    f1_ = f1_score(rna_label, rna_pred)
    precision_ = precision_score(rna_label, rna_pred)
    recall_ = recall_score(rna_label, rna_pred)
    # mcc_ = matthews_corrcoef(rna_label, np.where(rna_pred == 0, -1, rna_pred))
    logger.info(f"acc: {acc_:.4f}\nf1: {f1_:.4f}\nprecision: {precision_:.4f}\nrecall: {recall_:.4f}\n")

    protein_pred_test = (protein_cos_test>th_val[0])
    protein_id2conf = {}
    protein_id2conf_binary = {}
    all_performance = []
    all_performance_labels = []
    logger.info("\neach protein:")
    for i, id_ in id_mapping_test.items():
        mask_wo = protein_label_test[:, i]!=-1
        label_t = protein_label_test[:, i][mask_wo]

        """binary"""
        protein_id2conf_binary[id_] = confusion_matrix(label_t, binary_pred_test[mask_wo], labels=[0, 1])
        logger.info(f"{protein_id2conf_binary[id_]}")
        apr_ = average_precision_score(label_t, binary_pred_test_prob[:, 1][mask_wo])
        auc_ = roc_auc_score(label_t, binary_pred_test_prob[:, 1][mask_wo])
        logger.info(f"apr: {apr_}")
        logger.info(f"roc_auc: {auc_}")
        logger.info(classification_report(label_t, binary_pred_test[mask_wo], target_names=["rna 0", "rna 1"], digits=4))

        acc_ = accuracy_score(label_t, binary_pred_test[mask_wo])
        f1_ = f1_score(label_t, binary_pred_test[mask_wo])
        precision_ = precision_score(label_t, binary_pred_test[mask_wo])
        recall_ = recall_score(label_t, binary_pred_test[mask_wo])
        # mcc_ = matthews_corrcoef(label_t, np.where(binary_pred_test == 0, -1, binary_pred_test)[mask_wo])
        logger.info(f"acc: {acc_:.4f}\nf1: {f1_:.4f}\nprecision: {precision_:.4f}\nrecall: {recall_:.4f}\n")
        all_performance.append([apr_, auc_, acc_, precision_, recall_, f1_])
        all_performance_labels.append((label_t, binary_pred_test[mask_wo]))
    with open("/home/gaoyifei/RNARepresentation/a_refresh/res/ours_gat_10_9_labels.pkl", "wb") as f:
        pickle.dump(all_performance_labels, f)
    # with open("/home/gaoyifei/RNARepresentation/a_refresh/res/ours_gat_10_9.pkl", "wb") as f:
    #     pickle.dump(all_performance, f)
    print()
