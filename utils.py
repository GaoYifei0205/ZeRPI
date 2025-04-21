import os
import pickle
import scipy as sp
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import dgl
import argparse
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
import random


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


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=511)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--cutoff_cos', type=float, default=0.1)
    parser.add_argument('--save', type=int, default=0, choices=[0, 1])
    parser.add_argument('--just_test', type=int, default=1, choices=[0, 1])
    parser.add_argument('--use_binary_loss', type=int, default=1, choices=[0, 1])
    parser.add_argument('--use_binary_protein', type=int, default=1, choices=[0, 1])
    parser.add_argument('--use_cl_loss', type=int, default=1, choices=[0, 1])
    parser.add_argument('--model_save_dir', type=str, default="/data0/gaoyifei/leave_one_res/GCN")
    parser.add_argument('--device', type=int, default=0, choices=[i for i in range(-1, 8)])
    parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_reduce_factor', type=float, default=0.3)
    parser.add_argument('--lr_reduce_patience', type=int, default=100)
    parser.add_argument('--lr_type', type=str, default="reduce", choices=["reduce", "step", "cos"])
    parser.add_argument("--milestones", type=int, nargs='+', default=[20, 30])
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--cos_rate', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--use_lp_weight', type=int, default=0, choices=[0, 1])
    parser.add_argument('--lp_eps', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--loss_type', type=str, default="infonce", choices=["infonce", "cosine"])
    """just one protein or # leave one"""
    parser.add_argument('--protein_id', type=int, default=0, choices=[i for i in range(-1, 19)])
    """GCN"""
    parser.add_argument('--gnn', type=str, default="gcn", choices=["gcn", "gin", "sage", "gat"])
    parser.add_argument('--rna_in_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--batch_norm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gnn_n_layers', type=int, default=2)
    parser.add_argument('--rbp_feat_model', type=str, default="esm2_t33_650M_UR50D", 
                        choices=["esm1_t12_85M_UR50S", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D"])

    args = parser.parse_args()

    args.device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    args.use_lp_weight = True if args.use_lp_weight == 1 else False
    args.use_binary_loss = True if args.use_binary_loss == 1 else False
    args.use_binary_protein = True if args.use_binary_protein == 1 else False
    args.use_cl_loss = True if args.use_cl_loss == 1 else False
    args.just_test = True if args.just_test == 1 else False
    args.batch_norm = True if args.batch_norm == 1 else False
    return args


class MyDataset_raw(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["id"])

    def __getitem__(self, idx):
        return {"id": self.data["id"][idx], "label1": self.data["label1"][idx], "label2": self.data["label2"][idx], 
                "eps": self.data["eps"][idx], "use_lp_weight": self.data["use_lp_weight"][idx]}


# finetune either 1d or 3d, split the dataloader. memory?
def my_collate_fn(data):
    data_prefix = "/data0/gaoyifei/GraphProt_byRBP"
    batch_list = []
    label1s, label2s = [], []
    for one in data:
        id_ = one["id"]
        lp_eps = one["eps"]
        use_lp_weight = one["use_lp_weight"]
        p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", f"{use_lp_weight}_{lp_eps}")
        one_g, _ = load_graphs(os.path.join(p_, f"{id_}_dgl_graph.bin"))

        batch_list.append(one_g[0])
        label1s.append(one["label1"])
        label2s.append(one["label2"])
    return dgl.batch(batch_list), torch.as_tensor(label1s, dtype=torch.long), torch.as_tensor(np.array(label2s), dtype=torch.float32)

def my_collate_fn_multiple(data):
    data_prefix = "/data0/gaoyifei/GraphProt_byRBP"
    batch_list = []
    label1s, label2s = [], []
    for one in tqdm(data):
        id_ = one["id"]
        lp_eps = one["eps"]
        use_lp_weight = one["use_lp_weight"]
        p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", f"{use_lp_weight}_{lp_eps}", "multiple")
        one_g, _ = load_graphs(os.path.join(p_, f"{id_}_dgl_graph.bin"))

        batch_list.append(one_g[0])
        label1s.append(one["label1"])
        label2s.append(one["label2"])
    return dgl.batch(batch_list), torch.as_tensor(label1s, dtype=torch.long), torch.as_tensor(np.array(label2s), dtype=torch.float32)

def my_collate_fn_batch(data):
    data_prefix = "/amax/data/gaoyifei/GraphProt_byRBP"
    batch_list = []
    label1s, label2s = [], []
    for one in data:
        id_ = one["id"]
        lp_eps = one["eps"]
        use_lp_weight = one["use_lp_weight"]
        p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", f"{use_lp_weight}_{lp_eps}")
        one_g, _ = load_graphs(os.path.join(p_, f"{id_}_dgl_graph.bin"))

        batch_list.append(one_g[0])
        label1s.append(one["label1"])
        label2s.append(one["label2"])
    return batch_list, label1s, label2s


def my_collate_fn_512(data):
    batch_g, labels_dict = load_graphs(data[0])
    return dgl.batch(batch_g), labels_dict["label1"], labels_dict["label2"]
    # return batch_g[0], labels_dict["label1"], labels_dict["label2"]


class MyCollateFnOld:
    def __init__(self) -> None:
        data_prefix = "/data0/gaoyifei/GraphProt_byRBP"
        with open(os.path.join(data_prefix, "aa_RNA_data", "id2feat.pkl"), "rb") as f:
            self.id2feat =  pickle.load(f)

    def __call__(self, data):
        data_prefix = "/data0/gaoyifei/GraphProt_byRBP"
        batch_list = []
        label1s, label2s = [], []
        for one in tqdm(data):
            id_ = one["id"]
            lp_eps = one["eps"]
            use_lp_weight = one["use_lp_weight"]
            # folder_name = f"eps_{lp_eps}_weight_{use_lp_weight}"
            # save_path = os.path.join(data_prefix, "aa_processed_graphs", folder_name)
            # os.makedirs(save_path, exist_ok=True)

            matrix_ = sp.sparse.load_npz(os.path.join(data_prefix, "aa_RNA_data", "LP_matrix", f"{id_}.npz")).toarray()
            # add adj edges
            np.fill_diagonal(matrix_, 1)
            np.fill_diagonal(matrix_[1:], 1)
            np.fill_diagonal(matrix_[:,1:], 1)
            """stupid for!"""
            # for j in range(len(matrix_)):
            #     matrix_[j][j] = 1.
            #     if j+1 < len(matrix_):
            #         matrix_[j][j+1] = 1.
            #         matrix_[j+1][j] = 1.
            #     if j-1 > 0:
            #         matrix_[j-1][j] = 1.
            #         matrix_[j][j-1] = 1.

            # remove edges less than lp_eps
            matrix_[matrix_<one["eps"]] = 0.
            # feat_ = torch.load(os.path.join(data_prefix, "aa_RNA_data", "rna_feat", f"{id_}.pt")).squeeze()
            feat_ = torch.load(self.id2feat[id_]).squeeze()
            # dgl.from_scipy with sparse matrix; non-zero -> a bond
            if one["use_lp_weight"]:
                one_g = dgl.from_scipy(sp.sparse.csc_matrix(matrix_), eweight_name="edge_weight")
                one_g.edata["edge_weight"] = one_g.edata["edge_weight"].float()
            else:
                one_g = dgl.from_scipy(sp.sparse.csc_matrix(matrix_))
            one_g.ndata["h"] = feat_
            # one_g = dgl.add_self_loop(one_g)
            batch_list.append(one_g)
            label1s.append(one["label1"])
            label2s.append(one["label2"])
        # return dgl.batch(batch_list), torch.as_tensor(label1s, dtype=torch.long), torch.as_tensor(np.array(label2s), dtype=torch.float32)
        return batch_list, torch.as_tensor(label1s, dtype=torch.long), torch.as_tensor(np.array(label2s), dtype=torch.float32)


def load_fasta_format(file_path):
    all_id, all_seq = [], []
    with open(file_path, "r") as f:
        lines = f.readlines()
        seq = ''
        for row in lines:
            # if type(row) is bytes:
            #     row = row.decode('utf-8')
            row = row.rstrip()
            if row.startswith('>'):
                all_id.append(row)
                if seq != '':
                    all_seq.append(seq)
                    seq = ''
            else:
                seq += row
        all_seq.append(seq)
    return all_id, all_seq


"""save the lp matrix of each rna seq one by one, save the id mapping"""
def process_linearpartition(start_id=0, end_id=2):
    data_prefix = "/data0/gaoyifei/GraphProt_byRBP"
    """unique seq number"""
    with open(os.path.join(data_prefix, "aa_RNA_data", "mapping.pkl"), "rb") as f:
        mapping_ = pickle.load(f)
    rna2id, id2rna = mapping_["rna2id"], mapping_["id2rna"]

    # rna2id = {}
    # id2rna = {}
    # cnt = 0
    # # for each rbp folder
    # for rbp in rbps:
    #     # for each split
    #     for split_ in ["train", "test"]:
    #         # for each pos or neg
    #         for type_ in ["positives", "negatives"]:
    #             data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
    #             all_id, all_seq = load_fasta_format(data_path)
    #             for one in all_seq:
    #                 if one not in rna2id:
    #                     rna2id[one] = cnt
    #                     id2rna[cnt] = one
    #                     cnt += 1
    # with open(os.path.join(data_prefix, "aa_RNA_data", "mapping.pkl"), "wb") as f:
    #     pickle.dump({"rna2id": rna2id, "id2rna": id2rna}, f)
    # 1119292
    print(len(rna2id))
    print()

    processed_rna = set()
    # for each rbp folder
    for rbp in rbps[start_id:end_id]:
        # for each split
        for split_ in ["train", "test"]:
            # for each pos or neg
            for type_ in ["positives", "negatives"]:
                prob_path = os.path.join(data_prefix, rbp, split_, type_, "linearpartition_True_prob_mat.obj")
                if not os.path.exists(prob_path):
                    continue
                with open(prob_path, "rb") as f:
                    prob_ = pickle.load(f)

                # need sqrt!
                data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
                all_id, all_seq = load_fasta_format(data_path)
                for i, one_matrix in enumerate(tqdm(prob_)):
                    temp_ = one_matrix.toarray()
                    # remove RNA edges; add this one when creating dgl
                    for j in range(len(temp_)):
                        if j+1 < len(temp_):
                            temp_[j][j+1] = 0.
                            temp_[j+1][j] = 0.
                        if j-1 > 0:
                            temp_[j-1][j] = 0.
                            temp_[j][j-1] = 0.
                    temp_ = np.sqrt(temp_)
                    if all_seq[i] not in processed_rna:
                        id_ = rna2id[all_seq[i]]
                        sp.sparse.save_npz(os.path.join(data_prefix, "aa_RNA_data", "LP_matrix", f"{id_}.npz"), sp.sparse.csc_matrix(temp_))
                        processed_rna.add(all_seq[i])

                    # check symmetry and equality; do not consider tol
                    # assert (temp_==temp_.T).all()
                    # if all_seq[i] not in res_:
                    #     res_[all_seq[i]] = sp.sparse.csc_matrix(temp_)
                    # else:
                    #     assert not (sp.sparse.csc_matrix(temp_) != res_[all_seq[i]]).toarray().any()
                        # print(i)
                # too large! store one by one!
                # with open(os.path.join(data_prefix, "lp_res.pkl"), "wb") as f:
                #     pickle.dump(res_, f)
                # print()
    print()


"""save the 768 feat of each rna seq one by one, the id mapping"""
def process_rna_feat():
    data_prefix = "/data0/gaoyifei/GraphProt_byRBP"
    with open(os.path.join(data_prefix, "aa_RNA_data", "mapping.pkl"), "rb") as f:
        mapping_ = pickle.load(f)
    rna2id, id2rna = mapping_["rna2id"], mapping_["id2rna"]

    """just save the path to feat"""
    id2feat = {}
    processed_rna = set()
    # for each rbp folder
    for rbp in rbps:
        # for each split
        for split_ in ["train", "test"]:
            # for each pos or neg
            for type_ in ["positives", "negatives"]:
                rna_feat_path = os.path.join(data_prefix, rbp, split_, type_, "tensor")
                for root, dirs, files in os.walk(rna_feat_path):
                    temp_id = sorted([int(one[:-3]) for one in files])
                data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
                all_id, all_seq = load_fasta_format(data_path)
                for i, one_id in enumerate(tqdm(temp_id)):
                    rna_seq = all_seq[i]
                    if rna_seq not in processed_rna:
                        # e.g. [1, 340, 768]
                        # feat_ = torch.load(os.path.join(rna_feat_path, f"{one_id}.pt"))
                        # temp_p = os.path.join(data_prefix, "aa_RNA_data", "rna_feat", f"{rna2id[rna_seq]}.pt")
                        id2feat[rna2id[rna_seq]] = os.path.join(rna_feat_path, f"{one_id}.pt")
                        # if not os.path.exists(temp_p):
                        #     continue
                        # torch.save(feat_, temp_p)
                        processed_rna.add(rna_seq)
    with open(os.path.join(data_prefix, "aa_RNA_data", "id2feat.pkl"), "wb") as f:
        pickle.dump(id2feat, f)

    # processed_rna = set()
    # # for each rbp folder
    # for rbp in rbps:
    #     # for each split
    #     for split_ in ["train", "test"]:
    #         # for each pos or neg
    #         for type_ in ["positives", "negatives"]:
    #             rna_feat_path = os.path.join(data_prefix, rbp, split_, type_, "tensor")
    #             for root, dirs, files in os.walk(rna_feat_path):
    #                 temp_id = sorted([int(one[:-3]) for one in files])
    #             data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
    #             all_id, all_seq = load_fasta_format(data_path)
    #             for i, one_id in enumerate(tqdm(temp_id)):
    #                 rna_seq = all_seq[i]
    #                 if rna_seq not in processed_rna:
    #                     # e.g. [1, 340, 768]
    #                     feat_ = torch.load(os.path.join(rna_feat_path, f"{one_id}.pt"))
    #                     temp_p = os.path.join(data_prefix, "aa_RNA_data", "rna_feat", f"{rna2id[rna_seq]}.pt")
    #                     # if not os.path.exists(temp_p):
    #                     #     continue
    #                     torch.save(feat_, temp_p)
    #                     processed_rna.add(rna_seq)
    #             print()
    print()


"""save the rna-protein label of each rna seq one by one, 1(+), 0(-), -1(unknown), the id mapping"""
def process_label():
    id2rbp, rbp2id = {}, {}
    for i in range(len(rbps)):
        id2rbp[i] = rbps[i]
        rbp2id[rbps[i]] = i
    data_prefix = "/amax/data/gaoyifei/GraphProt_byRBP"
    with open(os.path.join(data_prefix, "aa_RNA_data", "mapping.pkl"), "rb") as f:
        mapping_ = pickle.load(f)
    rna2id, id2rna = mapping_["rna2id"], mapping_["id2rna"]
    rnaid2label1, rnaid2label2 = {}, {}
    for id_ in id2rna.keys():
        rnaid2label1[id_] = -1
        rnaid2label2[id_] = -np.ones((19,))

    processed_rna = set()
    # for each rbp folder
    for rbp in rbps:
        # for each split
        for split_ in ["train", "test"]:
            # for each pos or neg
            for type_ in ["positives", "negatives"]:
                data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
                all_id, all_seq = load_fasta_format(data_path)
                for i, one_rna_seq in enumerate(tqdm(all_seq)):
                    rnaid2label1[rna2id[one_rna_seq]] = 0 if (type_ == "negatives" and rnaid2label1[rna2id[one_rna_seq]] != 1) else 1
                    rnaid2label2[rna2id[one_rna_seq]][rbp2id[rbp]] = 0 if (type_ == "negatives" and rnaid2label2[rna2id[one_rna_seq]][rbp2id[rbp]] != 1) else 1
    with open(os.path.join(data_prefix, "aa_RNA_data", "mapping_label.pkl"), "wb") as f:
        pickle.dump({"id2label1": rnaid2label1, "id2label2": rnaid2label2}, f)
    print()


def process_graph_single():
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

    used_rbps = ['CAPR1', 'EWS', 'HNRPC', 'MOV10', 'PUM2', 'QKI', 'RTCB', 'SRSF1', 'TIA1', 'Z3H7B']
    used_ids = []
    for rbp in used_rbps:
        for split_ in ["train", "test"]:
            for type_ in ["positives", "negatives"]:
                data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
                _, train_seq = load_fasta_format(data_path)
                for one_seq in train_seq:
                    used_ids.append(rna2id[one_seq])

    for use_lp_weight in [False, True]:  # lp matrix weight
        for lp_eps in [0.1, 0.2, 0.3]:  # cutoff of the weight
            dataset_ = MyDataset_raw(used_ids)
            dataloader_ = DataLoader(dataset=dataset_, batch_size=64, shuffle=False, num_workers=16)
            for ids_ in tqdm(dataloader_):
                for id_ in ids_:
                    p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", f"{use_lp_weight}_{lp_eps}")
                    if not os.path.exists(p_):
                        os.makedirs(p_, exist_ok=False)

                    matrix_ = sp.sparse.load_npz(os.path.join(data_prefix, "aa_RNA_data", "LP_matrix", f"{id_}.npz")).toarray()
                    # add adj edges
                    np.fill_diagonal(matrix_, 1)
                    np.fill_diagonal(matrix_[1:], 1)
                    np.fill_diagonal(matrix_[:,1:], 1)

                    # remove edges less than lp_eps
                    matrix_[matrix_ < lp_eps] = 0.
                    feat_ = torch.load(os.path.join(data_prefix, "aa_RNA_data", "rna_feat", f"{id_}.pt")).squeeze()
                    # dgl.from_scipy with sparse matrix; non-zero -> a bond
                    if use_lp_weight:
                        one_g = dgl.from_scipy(sp.sparse.csc_matrix(matrix_), eweight_name="edge_weight")
                        one_g.edata["edge_weight"] = one_g.edata["edge_weight"].float()
                    else:
                        one_g = dgl.from_scipy(sp.sparse.csc_matrix(matrix_))
                    one_g.ndata["h"] = feat_
                    # print((one_g.num_edges()-one_g.num_nodes())/2/one_g.num_nodes())
                    save_graphs(os.path.join(p_, f"{id_}_dgl_graph.bin"), [one_g])
                    # print()
    

def process_graph_leave_one(start_id=0, end_id=2):
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
    with open(os.path.join(data_prefix, "aa_RNA_data", "id2feat.pkl"), "rb") as f:
        id2feat = pickle.load(f)

    # sample here? just 20000 for each
    used_ids = []
    for rbp in rbps[start_id:end_id]:
        for split_ in ["train", "test"]:
            for type_ in ["positives", "negatives"]:
                data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
                _, train_seq = load_fasta_format(data_path)
                random.shuffle(train_seq)
                for one_seq in train_seq[:10000]:
                    used_ids.append(rna2id[one_seq])

    for use_lp_weight in [False]:  # lp matrix weight
        for lp_eps in [0.1]:  # cutoff of the weight
            dataset_ = MyDataset_raw(used_ids)
            dataloader_ = DataLoader(dataset=dataset_, batch_size=128, shuffle=False, num_workers=16)
            for ids_ in tqdm(dataloader_):
                for id_ in tqdm(ids_):
                    p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", f"{use_lp_weight}_{lp_eps}")
                    if not os.path.exists(p_):
                        os.makedirs(p_, exist_ok=False)

                    matrix_ = sp.sparse.load_npz(os.path.join(data_prefix, "aa_RNA_data", "LP_matrix", f"{id_}.npz")).toarray()
                    # add adj edges
                    np.fill_diagonal(matrix_, 1)
                    np.fill_diagonal(matrix_[1:], 1)
                    np.fill_diagonal(matrix_[:,1:], 1)

                    # remove edges less than lp_eps
                    matrix_[matrix_ < lp_eps] = 0.
                    feat_ = torch.load(id2feat[int(id_)]).squeeze()
                    # dgl.from_scipy with sparse matrix; non-zero -> a bond
                    if use_lp_weight:
                        one_g = dgl.from_scipy(sp.sparse.csc_matrix(matrix_), eweight_name="edge_weight")
                        one_g.edata["edge_weight"] = one_g.edata["edge_weight"].float()
                    else:
                        one_g = dgl.from_scipy(sp.sparse.csc_matrix(matrix_))
                    one_g.ndata["h"] = feat_
                    # print((one_g.num_edges()-one_g.num_nodes())/2/one_g.num_nodes())
                    if not os.path.exists(os.path.join(p_, f"{id_}_dgl_graph.bin")):
                        save_graphs(os.path.join(p_, f"{id_}_dgl_graph.bin"), [one_g])
                    # print()
    

def process_batch():
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
    trainrbps = ['CAPR1', 'EWS', 'HNRPC', 'MOV10', 'PUM2', 'QKI', 'RTCB', 'SRSF1', 'TIA1', 'Z3H7B']
    testrbps = ['CAPR1', 'EWS', 'HNRPC', 'MOV10', 'PUM2', 'QKI', 'RTCB', 'SRSF1', 'TIA1', 'Z3H7B']
    # """add more"""
    # trainrbps = ['AGO2', 'ALKB5', 'FUS', 'NCBP3', 'PTBP1', 'RBP56', 'TADBP', 'TIAR']
    # testrbps = ['AGO2', 'ALKB5', 'FUS', 'NCBP3', 'PTBP1', 'RBP56', 'TADBP', 'TIAR']
    all_train_ids = set()
    all_test_ids = set()
    """sample 5000 for each type or not"""
    for temp_rbps, temp_type, temp_ids in zip([trainrbps, testrbps], ["train", "test"], [all_train_ids, all_test_ids]):
        for rbp in temp_rbps:
            for type_ in ["positives", "negatives"]:
                data_path = os.path.join(data_prefix, rbp, temp_type, type_, "data.fa")
                _, one_seqs = load_fasta_format(data_path)
                random.shuffle(one_seqs)
                temp_max_num = 0
                for one_ in one_seqs:
                    # if rna2id[one_] not in temp_ids:
                    #     temp_ids.add(rna2id[one_])
                    #     temp_max_num += 1
                    # if temp_max_num >= 5000:
                    #     break
                    temp_ids.add(rna2id[one_])
    all_train_ids = list(all_train_ids)
    random.shuffle(all_train_ids)
    all_test_ids = list(all_test_ids)
    random.shuffle(all_test_ids)
    print("load data done")
    # print(all_train_ids[:10], all_test_ids[:10])
    
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
    # print(train_dataset[0], val_dataset[0])


    # train_loader = DataLoader(dataset=train_dataset, batch_size=args_.batch_size, shuffle=False, 
    #                     collate_fn=my_collate_fn_batch, drop_last=False, num_workers=16)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=args_.batch_size, shuffle=False, 
    #                     collate_fn=my_collate_fn_batch, drop_last=False, num_workers=16)
    # testset = MyDataset(data=test_data_)
    # test_loader = DataLoader(dataset=testset, batch_size=args_.batch_size, shuffle=False, 
    #                         collate_fn=my_collate_fn_batch, drop_last=False, num_workers=16)
    
    # p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", "sub_10", f"{args_.use_lp_weight}_{args_.lp_eps}")
    # if not os.path.exists(p_):
    #     os.makedirs(p_)
    # for id_, (g, l1, l2) in enumerate(tqdm(train_loader)):
    #     save_graphs(filename=os.path.join(p_, f"training_batch_{id_}.bin"), g_list=g, labels={"label1": torch.as_tensor(np.array(l1)), "label2": torch.as_tensor(np.stack(l2)).float()})
    # for id_, (g, l1, l2) in enumerate(tqdm(val_loader)):
    #     save_graphs(filename=os.path.join(p_, f"validation_batch_{id_}.bin"), g_list=g, labels={"label1": torch.as_tensor(np.array(l1)), "label2": torch.as_tensor(np.stack(l2)).float()})
    # for id_, (g, l1, l2) in enumerate(tqdm(test_loader)):
    #     save_graphs(filename=os.path.join(p_, f"test_batch_{id_}.bin"), g_list=g, labels={"label1": torch.as_tensor(np.array(l1)), "label2": torch.as_tensor(np.stack(l2)).float()})
    # print()

    my_collate_fn_old = MyCollateFnOld()
    train_loader = DataLoader(dataset=train_dataset, batch_size=args_.batch_size, shuffle=False, 
                        collate_fn=my_collate_fn_old, drop_last=False, num_workers=16)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args_.batch_size, shuffle=False, 
                        collate_fn=my_collate_fn_old, drop_last=False, num_workers=16)
    testset = MyDataset(data=test_data_)
    test_loader = DataLoader(dataset=testset, batch_size=args_.batch_size, shuffle=False, 
                            collate_fn=my_collate_fn_old, drop_last=False, num_workers=16)
    
    p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", "sub_10", f"{args_.use_lp_weight}_{args_.lp_eps}")
    if not os.path.exists(p_):
        os.makedirs(p_)
    for id_, (g, l1, l2) in enumerate(tqdm(train_loader)):
        save_graphs(filename=os.path.join(p_, f"training_batch_{id_}.bin"), g_list=g, labels={"label1": l1, "label2": l2})
    for id_, (g, l1, l2) in enumerate(tqdm(val_loader)):
        save_graphs(filename=os.path.join(p_, f"validation_batch_{id_}.bin"), g_list=g, labels={"label1": l1, "label2": l2})
    for id_, (g, l1, l2) in enumerate(tqdm(test_loader)):
        save_graphs(filename=os.path.join(p_, f"test_batch_{id_}.bin"), g_list=g, labels={"label1": l1, "label2": l2})
    print()


def process_other_proteins():
    data_prefix = "/data0/gaoyifei/GraphProt_byRBP"
    id2rbp, rbp2id = {}, {}
    print("loading mapping")
    with open(os.path.join(data_prefix, "aa_RNA_data", "id2feat.pkl"), "rb") as f:
        id2feat =  pickle.load(f)
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
    args_.use_lp_weight = False  # lp matrix weight
    args_.lp_eps = 0.1  # cutoff of the weight
    seed_torch(args_.seed)

    print("loading data")
    all_zero_shot_ids = []
    zero_shot_rbps = rbps  # ['AGO2', 'ELAV1', 'ALKB5', 'FUS', 'NCBP3', 'PTBP1', 'RBP56', 'TADBP', 'TIAR']
    zero_shot_ids = [{"id": [], "label1": [], "label2": [], "eps": [], "use_lp_weight": []} for _ in zero_shot_rbps]
    protein_id = torch.as_tensor([rbp2id[one] for one in zero_shot_rbps])
    for i, rbp in enumerate(zero_shot_rbps):
        for split_ in ["test"]:
            for type_ in ["positives", "negatives"]:
                data_path = os.path.join(data_prefix, rbp, split_, type_, "data.fa")
                _, train_seq = load_fasta_format(data_path)
                for one_seq in train_seq:
                    id_ = rna2id[one_seq]
                    zero_shot_ids[i]["id"].append(id_)
                    zero_shot_ids[i]["eps"].append(args_.lp_eps)
                    zero_shot_ids[i]["use_lp_weight"].append(args_.use_lp_weight)
                    label1, label2 = rnaid2label1[id_], rnaid2label2[id_]
                    zero_shot_ids[i]["label1"].append(np.any(label2[protein_id]==1.))
                    zero_shot_ids[i]["label2"].append(label2[protein_id])

    p_ = os.path.join(data_prefix, "aa_RNA_data", "dgl_graphs", "others", f"{args_.use_lp_weight}_{args_.lp_eps}")
    if not os.path.exists(p_):
        os.makedirs(p_, exist_ok=False)
    for rbp_i in range(len(zero_shot_rbps)):
        dataset_ = MyDataset(zero_shot_ids[rbp_i])
        batch_list = []
        label1s, label2s, ids_list = [], [], []
        for one in tqdm(dataset_):
            id_ = one["id"]
            ids_list.append(id_)
            lp_eps = one["eps"]
            use_lp_weight = one["use_lp_weight"]

            matrix_ = sp.sparse.load_npz(os.path.join(data_prefix, "aa_RNA_data", "LP_matrix", f"{id_}.npz")).toarray()
            # add adj edges
            np.fill_diagonal(matrix_, 1)
            np.fill_diagonal(matrix_[1:], 1)
            np.fill_diagonal(matrix_[:,1:], 1)

            # remove edges less than lp_eps
            matrix_[matrix_ < args_.lp_eps] = 0.
            # feat_ = torch.load(os.path.join(data_prefix, "aa_RNA_data", "rna_feat", f"{id_}.pt")).squeeze()
            feat_ = torch.load(id2feat[id_]).squeeze()
            # dgl.from_scipy with sparse matrix; non-zero -> a bond
            if args_.use_lp_weight:
                one_g = dgl.from_scipy(sp.sparse.csc_matrix(matrix_), eweight_name="edge_weight")
                one_g.edata["edge_weight"] = one_g.edata["edge_weight"].float()
            else:
                one_g = dgl.from_scipy(sp.sparse.csc_matrix(matrix_))
            one_g.ndata["h"] = feat_

            batch_list.append(one_g)
            label1s.append(one["label1"])
            label2s.append(one["label2"])
        save_graphs(filename=os.path.join(p_, f"test_batch_{zero_shot_rbps[rbp_i]}.bin"), g_list=batch_list, labels={"label1": torch.as_tensor(np.array(label1s)), "label2": torch.as_tensor(np.stack(label2s)).float(), "label3": torch.as_tensor(ids_list)})
        print()
    print("load data done")


if __name__ == '__main__':
    # 10_10
    # process_batch()

    # 10_9
    # process_other_proteins()

    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=2)
    args = parser.parse_args()
    # args.end = args.start + 2
    print(args.start, args.end)
    process_linearpartition(args.start, args.end)
    # process_rna_feat()
    process_graph_leave_one(args.start, args.end)
    # process_label()

    print()
