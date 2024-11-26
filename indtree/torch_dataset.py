from torch.utils.data import Dataset
import torch
from indtree.coordinates import integer_bfs_index, binary_bfs_index
from typing import Literal


class TreeFitting(Dataset):
    def __init__(
        self,
        onehot_feat,
        onehot_class,
        onehot_thresh,
        coords_type: Literal["integer", "binary"] = "integer",
        normalize: bool = False,
    ):
        super().__init__()
        self.onehot_feat = torch.Tensor(onehot_feat)
        self.onehot_class = torch.Tensor(onehot_class)
        self.onehot_thresh = torch.Tensor(onehot_thresh)
        if coords_type == "integer":
            self.coords = torch.Tensor(
                integer_bfs_index(onehot_feat.shape[0])
            )  # .unsqueeze(dim=0).repeat(self.onehot_class.shape[0], 1, 1)
            if normalize:
                self.coords = self.coords / self.coords.max()
        elif coords_type == "binary":
            self.coords = torch.Tensor(
                binary_bfs_index(onehot_feat.shape[0])
            )  # .unsqueeze(dim=0).repeat(self.onehot_class.shape[0], 1, 1)
        else:
            raise ValueError(
                f"coords_type must be either 'integer' or 'binary', got {coords_type}"
            )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return self.coords, self.onehot_feat, self.onehot_class, self.onehot_thresh


class TreeFittingSplit(Dataset):
    def __init__(
        self,
        onehot_feat,
        onehot_class,
        onehot_thresh,
        coords_type: Literal["integer", "binary"] = "integer",
    ):
        super().__init__()
        self.onehot_feat = torch.Tensor(onehot_feat)
        self.onehot_class = torch.Tensor(onehot_class)
        self.onehot_thresh = torch.Tensor(onehot_thresh)
        if coords_type == "integer":
            self.coords = torch.Tensor(
                integer_bfs_index(onehot_feat.shape[0])
            )  # .unsqueeze(dim=0).repeat(self.onehot_class.shape[0], 1, 1)
        elif coords_type == "binary":
            self.coords = torch.Tensor(
                binary_bfs_index(onehot_feat.shape[0])
            )  # .unsqueeze(dim=0).repeat(self.onehot_class.shape[0], 1, 1)
        else:
            raise ValueError(
                f"coords_type must be either 'integer' or 'binary', got {coords_type}"
            )
        assert (
            len(self.onehot_feat) == len(self.onehot_class) == len(self.onehot_thresh)
        ), (f"Lengths of onehot_feat, onehot_class, and onehot_thresh must be equal, "
            f"got {len(self.onehot_feat)}, {len(self.onehot_class)}, {len(self.onehot_thresh)}")
        self.length = self.onehot_class.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (
            self.coords[idx],
            self.onehot_feat[idx],
            self.onehot_class[idx],
            self.onehot_thresh[idx],
        )


class TreeFittingRegression(Dataset):
    def __init__(
        self,
        vec_feat,
        vec_thresh,
        vec_class,
        coords_type: Literal["integer", "binary"] = "integer",
    ):
        super().__init__()
        self.vec_feat = torch.Tensor(vec_feat)
        self.vec_thresh = torch.Tensor(vec_thresh)
        self.vec_class = torch.Tensor(vec_class)
        if coords_type == "integer":
            self.coords = torch.Tensor(
                integer_bfs_index(vec_feat.shape[0])
            )  # .unsqueeze(dim=0).repeat(self.onehot_class.shape[0], 1, 1)
        elif coords_type == "binary":
            self.coords = torch.Tensor(
                binary_bfs_index(vec_feat.shape[0])
            )  # .unsqueeze(dim=0).repeat(self.onehot_class.shape[0], 1, 1)
        else:
            raise ValueError(
                f"coords_type must be either 'integer' or 'binary', got {coords_type}"
            )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return self.coords, self.vec_feat, self.vec_thresh, self.vec_class
