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


if __name__ == '__main__':
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
    args_.gnn = "gin"
    print(args_.model_save_dir)
    seed_torch(args_.seed)

    print("loading data")
    def load_old():
        trainrbps = ['CAPR1', 'EWS', 'HNRPC', 'MOV10', 'PUM2', 'QKI', 'RTCB', 'SRSF1', 'TIA1', 'Z3H7B']
        testrbps = ['CAPR1', 'EWS', 'HNRPC', 'MOV10', 'PUM2', 'QKI', 'RTCB', 'SRSF1', 'TIA1', 'Z3H7B']
        all_train_ids = []
        all_test_ids = []
        for rbp in trainrbps:
            for split_ in ["train"]:
                for type_ in ["positives", "negatives"]:
                    data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
                    _, train_seq = load_fasta_format(data_path)
                    for one_seq in train_seq:
                        all_train_ids.append(rna2id[one_seq])
        temp_id = 30000 if args_.just_test else 10000000
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

        protein_id = torch.as_tensor([rbp2id[one] for one in testrbps])
        for id_ in all_train_ids:
            train_data_["id"].append(id_)
            train_data_["eps"].append(args_.lp_eps)
            train_data_["use_lp_weight"].append(args_.use_lp_weight)
            label1, label2 = rnaid2label1[id_], rnaid2label2[id_]
            """turn 19 labels to specific label"""
            train_data_["label1"].append(np.any(label2[protein_id]==1.))
            train_data_["label2"].append(label2[protein_id])
        for id_ in all_test_ids:
            test_data_["id"].append(id_)
            test_data_["eps"].append(args_.lp_eps)
            test_data_["use_lp_weight"].append(args_.use_lp_weight)
            label1, label2 = rnaid2label1[id_], rnaid2label2[id_]
            test_data_["label1"].append(np.any(label2[protein_id]==1.))
            test_data_["label2"].append(label2[protein_id])
        trainset = MyDataset(data=train_data_)

        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
        train_loader = DataLoader(dataset=train_dataset, batch_size=args_.batch_size, shuffle=True, 
                            collate_fn=my_collate_fn, drop_last=False, num_workers=16)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args_.batch_size, shuffle=True, 
                            collate_fn=my_collate_fn, drop_last=False, num_workers=16)
        testset = MyDataset(data=test_data_)
        test_loader = DataLoader(dataset=testset, batch_size=args_.batch_size, shuffle=True, 
                                collate_fn=my_collate_fn, drop_last=False, num_workers=16)
    trainrbps = ['CAPR1', 'EWS', 'HNRPC', 'MOV10', 'PUM2', 'QKI', 'RTCB', 'SRSF1', 'TIA1', 'Z3H7B']    
    testrbps = ['CAPR1', 'EWS', 'HNRPC', 'MOV10', 'PUM2', 'QKI', 'RTCB', 'SRSF1', 'TIA1', 'Z3H7B']
    protein_id = torch.as_tensor([rbp2id[one] for one in testrbps])    
    train_dataset, val_dataset, test_dataset = [], [], []
    # "dgl_graphs/sub_10"
    p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", "sub_10", f"{args_.use_lp_weight}_{args_.lp_eps}")
    for type_, oneset in zip(["training", "validation", "test"], [train_dataset, val_dataset, test_dataset]):
        for id_ in range(500):
            temp_p = os.path.join(p_, f"{type_}_batch_{id_}.bin")
            if os.path.exists(temp_p):
                oneset.append(temp_p)
            else:
                break
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, collate_fn=my_collate_fn_512, shuffle=False, drop_last=False, num_workers=8)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, collate_fn=my_collate_fn_512, shuffle=False, drop_last=False, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, collate_fn=my_collate_fn_512, shuffle=False, drop_last=False, num_workers=8)
    
    print(f"trainset: {len(train_dataset)}, valset: {len(val_dataset)}, testset: {len(test_dataset)}")
    # out_dir = "/home/gaoyifei/RNARepresentation/out/GCN/"
    # TRAINSET_NAME = '_'.join(trainrbps)
    # root_log_dir = out_dir + 'logs/' + TRAINSET_NAME + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    # write_config_file = out_dir + 'configs/config_' + TRAINSET_NAME + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    
    # log_dir = os.path.join(root_log_dir)
    # print("log_dir: ", log_dir)
    # writer = SummaryWriter(log_dir=log_dir)

    # one_gcn = GCNLayer(768, 512)
    trainer_ = Trainer(args=args_, model=GNNNet(args=args_))

    best_p = [100 for _ in range(3)]
    training_step, val_step, test_step = 0, 0, 0
    for epoch in range(1, args_.epochs + 1):
        training_loss, training_t, training_step, training_conf_1, training_conf_2 = trainer_.training(train_loader, epoch, training_step, protein_id, "training", eval_every=100)
        val_loss, val_t, val_step, val_conf_1, val_conf_2 = trainer_.training(val_loader, epoch, val_step, protein_id, "validation", eval_every=50)
        test_loss, test_t, test_step, test_conf_1, test_conf_2 = trainer_.training(test_loader, epoch, test_step, protein_id, "test")
        trainer_.scheduler.step(metrics=val_loss)
        trainer_.logger.info("epoch {}, loss training: {:.6f}, loss val: {:.6f}, loss test: {:.6f}, time: {:.2f}".
                             format(epoch, training_loss, val_loss, test_loss, training_t + val_t + test_t))
        trainer_.logger.info(f"conf 1: {training_conf_1}, conf 2: {training_conf_2}\nconf 1: {val_conf_1}, conf 2: {val_conf_2}\nconf 1: {test_conf_1}, conf 2: {test_conf_2}\n")
        is_best = val_loss < best_p[1]
        if val_loss < best_p[1]:
            best_p = [training_loss, val_loss, test_loss]
        if args_.save == 1:
            trainer_.save(epoch, best_p, is_best, args_.model_save_dir)
    # RMSE, MAE, R2
    trainer_.logger.info("train/val/test:\n{}".format(best_p))
    if args_.save == 1:
        trainer_.writer.close()

    print()
