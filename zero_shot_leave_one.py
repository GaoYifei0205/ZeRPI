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
    with open(os.path.join(data_prefix, "aa_RNA_data", "mapping.pkl"), "rb") as f:
        mapping_ = pickle.load(f)
    rna2id, id2rna = mapping_["rna2id"], mapping_["id2rna"]
    with open(os.path.join(data_prefix, "aa_RNA_data", "mapping_label.pkl"), "rb") as f:
        mapping_label = pickle.load(f)
    rnaid2label1, rnaid2label2 = mapping_label["id2label1"], mapping_label["id2label2"]
    args_ = get_args()
    args_.save = 0
    args_.use_binary_protein = True
    # args_.use_lp_weight = False
    # args_.lp_eps = 0.1
    args_.epochs = 50
    args_.device = "cuda:4"
    args_.rbp_feat_model = "esm2_t33_650M_UR50D"
    seed_torch(args_.seed)

    print("loading data")
    args_.just_test = 0
    for kk in range(19):
        args_.protein_id = kk
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
        # print("load data done")
            
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


    all_performance = []
    for k in range(len(rbps)):
        one_performance = [-1 for _ in range(6)]
        for save_type in ["loss_best", "bi_auc"]:
            dir_prefix = f"512_0.001_False_0.1_30_gcn_True_True_True_{save_type}"
            log_path = os.path.join("/home/gaoyifei/RNARepresentation/out/GCN/zero_shot", dir_prefix+".txt")
            logger = create_file_logger(log_path)

            zero_shot_rbps = [rbps[k]]
            protein_id = torch.as_tensor([rbp2id[one] for one in zero_shot_rbps])    
            zero_shot_dataset = []
            p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", "others", f"{args_.use_lp_weight}_{args_.lp_eps}")
            zero_shot_dataloader = []
            for rbp_i in range(len(zero_shot_rbps)):
                temp_p = os.path.join(p_, f"test_batch_{zero_shot_rbps[rbp_i]}.bin")
                batch_g, labels_dict = load_graphs(temp_p)
                zero_shot_dataloader.append([dgl.batch(batch_g), labels_dict["label1"], labels_dict["label2"][:, protein_id]])
            
            model = GNNNet(args=args_)
            model_prefix = "/home/gaoyifei/RNARepresentation/out/GCN/models"
            p_ = os.path.join("/data0/gaoyifei/leave_one_res/GCN/models", f"{k}_{dir_prefix}.pth")
            if not os.path.exists(p_):
                print(f"The model path '{p_}' does not exist.")
                exit()
            model.load_state_dict(torch.load(p_)["model_state_dict"])
            logger.info(f"loading pretrained weights from {p_}")
            trainer_ = Trainer(args=args_, model=model)

            th_val = [0.]
            pos_test, neg_test, th_test, protein_cos_test, protein_label_test, id_mapping_test, binary_label_test, binary_pred_test_prob = trainer_.get_similarity(zero_shot_dataloader, protein_id)
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

            protein_pred_test = (protein_cos_test>th_val[0])
            protein_id2conf = {}
            protein_id2conf_binary = {}
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

                if auc_ > one_performance[1]:
                    one_performance = [apr_, auc_, acc_, precision_, recall_, f1_]
        all_performance.append(one_performance)
    with open("/home/gaoyifei/RNARepresentation/a_refresh/res/ours_leave_one.pkl", "wb") as f:
        pickle.dump(all_performance, f)
    print(np.array(all_performance))
    print()
