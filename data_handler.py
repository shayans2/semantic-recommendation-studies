import os

import numpy as np

import torch as t
import torch.utils.data as data

import pickle

import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix

from utils import get_device


class DataHandler:
    def __init__(
        self,
        user_num,
        item_num,
        dataset_name,
        test_batch_size=1024,
        train_batch_size=1024,
        device=None,
    ):
        self.user_num = user_num
        self.item_num = item_num
        self.device = get_device() if device is None else device
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.item_sem_emb = None
        self.user_sem_emb = None

        self.item_sim_emb = None
        self.user_sim_emb = None

        # Extra modality features consumed by the LATTICE variant. No dataset in
        # ./data ships these yet, so they stay None unless assigned by the caller.
        self.item_image_feats = None
        self.item_text_feats = None

        self.dataset_name = dataset_name

    def _load_one_mat(self, file):
        """Load one single adjacent matrix from file

        Args:
            file (string): path of the file to load

        Returns:
            scipy.sparse.coo_matrix: the loaded adjacent matrix
        """
        with open(file, "rb") as fs:
            mat = (pickle.load(fs) != 0).astype(np.float32)
        if type(mat) != coo_matrix:
            mat = coo_matrix(mat)
        return mat

    def _normalize_adj(self, mat):
        """
        Laplacian normalization for adjacency matrix in COO format.

        Args:
            mat (scipy.sparse.coo_matrix): The un-normalized adjacency matrix.

        Returns:
            scipy.sparse.coo_matrix: The normalized adjacency matrix.
        """
        degree = np.array(
            mat.sum(axis=-1)
        ).flatten()  # Get the sum of each row (degree of each node)

        # Inverse square root of the degree matrix, setting entries with degree=0 to 0
        d_inv_sqrt = np.power(
            degree, -0.5, where=degree > 0
        )  # Apply only where degree > 0
        d_inv_sqrt[degree == 0] = (
            0  # For entries with degree 0, set to 0 to avoid division by zero
        )

        # Create the diagonal matrix from the inverse square root degree matrix
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)

        # Apply the normalization
        return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

    def _make_torch_adj(self, mat, self_loop=False):
        """Transform uni-directional adjacent matrix in coo_matrix into bi-directional adjacent matrix in torch.sparse.FloatTensor

        Args:
            mat (coo_matrix): the uni-directional adjacent matrix

        Returns:
            torch.sparse.FloatTensor: the bi-directional matrix
        """
        if not self_loop:
            a = csr_matrix((self.user_num, self.user_num))
            b = csr_matrix((self.item_num, self.item_num))
        else:
            data = np.ones(self.user_num)
            row_indices = np.arange(self.user_num)
            column_indices = np.arange(self.user_num)
            a = csr_matrix(
                (data, (row_indices, column_indices)),
                shape=(self.user_num, self.user_num),
            )

            data = np.ones(self.item_num)
            row_indices = np.arange(self.item_num)
            column_indices = np.arange(self.item_num)
            b = csr_matrix(
                (data, (row_indices, column_indices)),
                shape=(self.item_num, self.item_num),
            )

        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = self._normalize_adj(mat)

        # make torch tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse_coo_tensor(idxs, vals, shape).to(self.device)

    def _load_optional(self, filename):
        """Load a side-information pickle if it exists, otherwise return None.

        Only some models use these, and not every dataset ships them, so a
        missing file must not stop a plain LightGCN run.
        """
        path = os.path.join("./data", self.dataset_name, filename)
        if not os.path.exists(path):
            print(f"[DataHandler] optional file not found, skipping: {path}")
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _align_side_embeds(self, user_emb, item_emb, kind):
        """Check side embeddings line up with the interaction matrix.

        The yelp `*_mm_emb_np.pkl` pair ships with the user and item files
        swapped. Detect that exact case and put them back, instead of failing
        later inside a model with an opaque shape error.
        """
        if user_emb is None or item_emb is None:
            return user_emb, item_emb

        if user_emb.shape[0] == self.user_num and item_emb.shape[0] == self.item_num:
            return user_emb, item_emb

        if user_emb.shape[0] == self.item_num and item_emb.shape[0] == self.user_num:
            print(
                f"[DataHandler] WARNING: {kind} embeddings for "
                f"'{self.dataset_name}' are swapped on disk (user file has "
                f"{user_emb.shape[0]} rows, item file has {item_emb.shape[0]}, "
                f"expected {self.user_num}/{self.item_num}). Swapping them back."
            )
            return item_emb, user_emb

        raise ValueError(
            f"{kind} embeddings for '{self.dataset_name}' do not match the "
            f"interaction matrix: got user={user_emb.shape[0]} rows and "
            f"item={item_emb.shape[0]} rows, expected {self.user_num} and "
            f"{self.item_num}."
        )

    def load_data(self):
        trn_mat = self._load_one_mat(self.trn_file)
        val_mat = self._load_one_mat(self.val_file)
        tst_mat = self._load_one_mat(self.tst_file)

        user_embeds_np = self._load_optional("usr_emb_np.pkl")
        item_embeds_np = self._load_optional("itm_emb_np.pkl")
        user_sim_emb = self._load_optional("usr_mm_emb_np.pkl")
        item_sim_emb = self._load_optional("itm_mm_emb_np.pkl")

        self.trn_mat = trn_mat
        self.user_num, self.item_num = trn_mat.shape

        # Validate against the interaction matrix now that the true counts are known.
        user_embeds_np, item_embeds_np = self._align_side_embeds(
            user_embeds_np, item_embeds_np, "semantic (*_emb_np)"
        )
        user_sim_emb, item_sim_emb = self._align_side_embeds(
            user_sim_emb, item_sim_emb, "multimodal (*_mm_emb_np)"
        )

        self.item_sem_emb = item_embeds_np
        self.user_sem_emb = user_embeds_np

        self.item_sim_emb = item_sim_emb
        self.user_sim_emb = user_sim_emb
        self.torch_adj = self._make_torch_adj(trn_mat)

        trn_data = PairwiseTrnData(trn_mat, self.item_num)

        val_data = AllRankTstData(val_mat, trn_mat)
        tst_data = AllRankTstData(tst_mat, trn_mat)
        self.test_dataloader = data.DataLoader(
            tst_data, batch_size=self.test_batch_size, shuffle=False, num_workers=0
        )
        self.valid_dataloader = data.DataLoader(
            val_data, batch_size=self.test_batch_size, shuffle=False, num_workers=0
        )
        self.train_dataloader = data.DataLoader(
            trn_data, batch_size=self.train_batch_size, shuffle=True, num_workers=0
        )


class PairwiseTrnData(data.Dataset):
    def __init__(self, coomat, item_num):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        self.item_num = item_num

    def sample_negs(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(self.item_num)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]


class AllRankTstData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0

        user_pos_lists = [list() for i in range(coomat.shape[0])]
        # user_pos_lists = set()
        test_users = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            user_pos_lists[row].append(col)
            test_users.add(row)
        self.test_users = np.array(list(test_users))
        self.user_pos_lists = user_pos_lists

    def __len__(self):
        return len(self.test_users)

    def __getitem__(self, idx):
        pck_user = self.test_users[idx]
        pck_mask = self.csrmat[pck_user].toarray()
        pck_mask = np.reshape(pck_mask, [-1])
        return pck_user, pck_mask
