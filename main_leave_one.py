import os
import pickle
import numpy as np
from utils import *
from model import *
import scipy as sp
from tqdm import tqdm
import dgl
import torch
# from dgl import save_graphs, load_graphs
# from torch.utils.data import Dataset, DataLoader, random_split
# from utils import my_collate_fn, load_fasta_format, get_args, MyDataset
import time
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score


if __name__ == '__main__':
    rbps = [
        'AGO2', 
        # 'AGO1234', 
        'ALKB5', 
        'CAPR1', 
        'ELAV1', 
        'EWS', 
        'FUS', 
        'HNRPC',
        # 'IGF2BP123', 
        'MOV10', 
        'NCBP3', 
        'PTBP1', 
        'PUM2', 
        'QKI', 
        'RBP56', 
        'RTCB',
        'SRSF1', 
        'TADBP', 
        'TIA1', 
        'TIAR', 
        'Z3H7B'
    ]

    data_prefix = "/data0/gaoyifei/GraphProt_byRBP"
    id2rbp, rbp2id = {}, {}
    print("loading mapping")
    for i in range(len(rbps)):
        id2rbp[i] = rbps[i]
        rbp2id[rbps[i]] = i
    with open(os.path.join(data_prefix, "aa_RNA_data", "mapping.pkl"), "rb") as f:
        mapping_ = pickle.load(f)
    rna2id, id2rna = mapping_["rna2id"], mapping_["id2rna"]
    with open(os.path.join(data_prefix, "aa_RNA_data", "mapping_label.pkl"), "rb") as f:
        mapping_label = pickle.load(f)
    rnaid2label1, rnaid2label2 = mapping_label["id2label1"], mapping_label["id2label2"]
    print("load mapping done")
    args_ = get_args()
    seed_torch(args_.seed)

    print("loading data")
    trainrbps = [rbps[i] for i in range(len(rbps)) if i != args_.protein_id]
    testrbps = [rbps[args_.protein_id]]
    all_train_ids = []
    all_test_ids = []
    # sample, 18vs1, 20000 at most?
    for rbp in trainrbps:
        for split_ in ["train"]:
            for type_ in ["positives", "negatives"]:
                data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
                _, train_seq = load_fasta_format(data_path)
                random.shuffle(train_seq)
                for one_seq in train_seq:
                    id_ = rna2id[one_seq]
                    p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", f"{args_.use_lp_weight}_{args_.lp_eps}")
                    if os.path.exists(os.path.join(p_, f"{id_}_dgl_graph.bin")):
                        all_train_ids.append(id_)
    temp_id = 20000 if args_.just_test else 10000000
    all_train_ids = list(set(all_train_ids))
    random.shuffle(all_train_ids)
    all_train_ids = all_train_ids[:temp_id]
    for rbp in testrbps:
        for split_ in ["test"]:
            for type_ in ["positives", "negatives"]:
                data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
                _, test_seq = load_fasta_format(data_path)
                for one_seq in test_seq:
                    all_test_ids.append(rna2id[one_seq])
    temp_id = 2000 if args_.just_test else 10000000
    all_test_ids = list(set(all_test_ids))
    random.shuffle(all_test_ids)
    all_test_ids = all_test_ids[:temp_id]
    print("load data done")
        
    # use_lp_weight = False  # lp matrix weight
    # lp_eps = 0.1  # cutoff of the weight
    train_data_ = {"id": [], "label1": [], "label2": [], "eps": [], "use_lp_weight": []}
    test_data_ = {"id": [], "label1": [], "label2": [], "eps": [], "use_lp_weight": []}

    # protein_id = torch.as_tensor([rbp2id[one] for one in testrbps])

    protein_id_train = torch.as_tensor([rbp2id[one] for one in trainrbps])
    for id_ in all_train_ids:
        train_data_["id"].append(id_)
        train_data_["eps"].append(args_.lp_eps)
        train_data_["use_lp_weight"].append(args_.use_lp_weight)
        label1, label2 = rnaid2label1[id_], rnaid2label2[id_]
        """turn 19 labels to specific label"""
        train_data_["label1"].append(np.any(label2[protein_id_train]==1.))
        # just one id, shape change!
        train_data_["label2"].append(label2[protein_id_train] if protein_id_train.shape[0]>1.5 else np.array([label2[protein_id_train]]))
    protein_id_test = torch.as_tensor([rbp2id[one] for one in testrbps])
    for id_ in all_test_ids:
        test_data_["id"].append(id_)
        test_data_["eps"].append(args_.lp_eps)
        test_data_["use_lp_weight"].append(args_.use_lp_weight)
        label1, label2 = rnaid2label1[id_], rnaid2label2[id_]
        test_data_["label1"].append(np.any(label2[protein_id_test]==1.))
        # just one id, shape change!
        test_data_["label2"].append(label2[protein_id_test] if protein_id_test.shape[0]>1.5 else np.array([label2[protein_id_test]]))
    trainset = MyDataset(data=train_data_)

    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args_.batch_size, shuffle=True, 
                        collate_fn=my_collate_fn, drop_last=False, num_workers=32)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args_.batch_size, shuffle=True, 
                        collate_fn=my_collate_fn, drop_last=False, num_workers=32)
    testset = MyDataset(data=test_data_)
    test_loader = DataLoader(dataset=testset, batch_size=args_.batch_size, shuffle=True, 
                            collate_fn=my_collate_fn, drop_last=False, num_workers=32)

    print(f"trainset: {len(train_dataset)}, valset: {len(val_dataset)}, testset: {len(testset)}")
    model = GNNNet(args=args_)
    trainer_ = Trainer(args=args_, model=model)
    
    dir_prefix = f"{args_.protein_id}_{args_.batch_size}_{args_.lr}_{args_.use_lp_weight}_{args_.lp_eps}_{args_.epochs}_{args_.gnn}_{args_.use_cl_loss}_{args_.use_binary_loss}_{args_.use_binary_protein}"
    log_path = os.path.join("/data0/gaoyifei/leave_one_res/GCN/res", dir_prefix+".txt")
    logger = create_file_logger(log_path)
    print()


    """loss, auc, ap"""
    best_p = [100 for _ in range(3)]
    best_cos_auc, best_cos_apr = [0, 0, "", ""], [0, 0, "", ""]
    best_bi_auc, best_bi_apr = [0, 0, "", ""], [0, 0, "", ""]
    
    training_step, val_step, test_step = 0, 0, 0
    for epoch in range(1, args_.epochs + 1):
        training_loss, training_t, training_step, training_conf_1, training_conf_2 = trainer_.training(train_loader, epoch, training_step, protein_id_train, "training", eval_every=50)
        val_loss, val_t, val_step, val_conf_1, val_conf_2 = trainer_.training(val_loader, epoch, val_step, protein_id_train, "validation", eval_every=10)
        test_loss, test_t, test_step, test_conf_1, test_conf_2 = trainer_.training(test_loader, epoch, test_step, protein_id_test, "test")
        trainer_.scheduler.step(metrics=val_loss)
        trainer_.logger.info("epoch {}, loss training: {:.6f}, loss val: {:.6f}, loss test: {:.6f}, time: {:.2f}".
                             format(epoch, training_loss, val_loss, test_loss, training_t + val_t + test_t))
        trainer_.logger.info(f"conf 1: {training_conf_1}, conf 2: {training_conf_2}\nconf 1: {val_conf_1}, conf 2: {val_conf_2}\nconf 1: {test_conf_1}, conf 2: {test_conf_2}\n")

        is_best = val_loss < best_p[1]
        if is_best:
            best_p = [training_loss, val_loss, test_loss]
        if args_.save == 1:
            trainer_.save(epoch, best_p, is_best, args_.model_save_dir, "_loss")

        th_val = [0.]
        pos_test, neg_test, th_test, protein_cos_test, protein_label_test, id_mapping_test, binary_label_test, binary_pred_test_prob = trainer_.get_similarity(test_loader, protein_id_test)
        protein_pred_test = (protein_cos_test>th_val[0])
        binary_pred_test = np.argmax(binary_pred_test_prob, axis=1)
        conf_matrix_binary = confusion_matrix(binary_label_test, binary_pred_test, labels=[0, 1])
        rna_label = np.concatenate([np.ones(pos_test.shape), np.zeros(neg_test.shape)], axis=0)
        rna_temp = np.concatenate([pos_test, neg_test], axis=0)
        rna_pred = (rna_temp>th_val[0])
        rna_auc = roc_auc_score(rna_label, (rna_temp+1)/2)
        conf_matrix_rna = confusion_matrix(rna_label, rna_pred, labels=[0, 1])
        protein_id2conf_binary = {}
        for i, id_ in id_mapping_test.items():

            mask_wo = protein_label_test[:, i]!=-1
            label_t = protein_label_test[:, i][mask_wo]

            """binary"""
            protein_id2conf_binary[id_] = confusion_matrix(label_t, binary_pred_test[mask_wo], labels=[0, 1])
            bi_roc_auc = roc_auc_score(label_t, binary_pred_test_prob[:, 1][mask_wo])
            bi_apr = average_precision_score(label_t, binary_pred_test_prob[:, 1][mask_wo])
            bi_cls_report = classification_report(label_t, binary_pred_test[mask_wo], target_names=["rna 0", "rna 1"], digits=4)
            if args_.save == 1:
                is_best = bi_roc_auc > best_bi_auc[0]
                trainer_.save(epoch, best_p, is_best, args_.model_save_dir, "_bi_auc")
                is_best = bi_apr > best_bi_apr[1]
                trainer_.save(epoch, best_p, is_best, args_.model_save_dir, "_bi_apr")
            if bi_roc_auc > best_bi_auc[0]:
                best_bi_auc = [bi_roc_auc, bi_apr, bi_cls_report, conf_matrix_binary]
            if bi_apr > best_bi_apr[1]:
                best_bi_apr = [bi_roc_auc, bi_apr, bi_cls_report, conf_matrix_binary]

    # RMSE, MAE, R2
    trainer_.logger.info("train/val/test:\n{}".format(best_p))
    logger.info("best_bi_auc")
    for i in range(4):
        logger.info(best_bi_auc[i])
    logger.info("best_bi_apr")
    for i in range(4):
        logger.info(best_bi_apr[i])
    if args_.save == 1:
        trainer_.writer.close()

    print()
