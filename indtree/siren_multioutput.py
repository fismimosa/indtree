import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from indtree.siren_models import SineLayer

DICT_MULTIOUTPUT_MODELS = {}


def register_model(name):
    def decorator(cls):
        DICT_MULTIOUTPUT_MODELS[name] = cls
        return cls

    return decorator


@register_model("siren")
class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        out_classes,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)
        self.head1 = nn.Linear(hidden_features, out_features)
        self.head2 = nn.Linear(hidden_features, out_classes)
        self.head3 = nn.Linear(hidden_features, 1)
        with torch.no_grad():  # this one I copied from the original code
            self.head3.weight.uniform_(
                -np.sqrt(6 / hidden_features) / hidden_omega_0,
                np.sqrt(6 / hidden_features) / hidden_omega_0,
            )

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        h = self.net(coords)

        out_features = self.head1(h)
        out_classes = self.head2(h)
        out_thresholds = self.head3(h)

        out_features = F.softmax(out_features, dim=-1)
        out_classes = F.softmax(out_classes, dim=-1)

        return out_features, out_classes, out_thresholds, coords


@register_model("siren_multi3rows")
class SirenMulti3Rows(Siren):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        out_classes,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__(
            in_features,
            hidden_features,
            hidden_layers,
            1,
            1,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        h = self.net(coords)

        out_features = self.head1(h)  # .int()
        out_classes = self.head2(h)  # .int()
        out_thresholds = self.head3(h)

        # out_features = F.softmax(out_features, dim=-1)
        # out_classes = F.softmax(out_classes, dim=-1)

        return out_features, out_classes, out_thresholds, coords


@register_model("siren_single_branch")
class SirenSingleBranch(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        out_classes,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net_feat = []
        self.net_class = []
        self.net_thresh = []

        for net, branch in zip(
            [self.net_feat, self.net_class, self.net_thresh],
            ["feat", "class", "thresh"],
        ):
            net.append(
                SineLayer(
                    in_features, hidden_features, is_first=True, omega_0=first_omega_0
                )
            )

            for _ in range(hidden_layers):
                net.append(
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        is_first=False,
                        omega_0=hidden_omega_0,
                    )
                )

            if branch == "feat":
                self.net_feat = nn.Sequential(*net)
            elif branch == "class":
                self.net_class = nn.Sequential(*net)
            elif branch == "thresh":
                self.net_thresh = nn.Sequential(*net)

        self.head_feat = nn.Linear(hidden_features, out_features)
        self.head_class = nn.Linear(hidden_features, out_classes)
        self.head_thresh = nn.Linear(hidden_features, 1)
        with torch.no_grad():  # this one I copied from the original code
            self.head_thresh.weight.uniform_(
                -np.sqrt(6 / hidden_features) / hidden_omega_0,
                np.sqrt(6 / hidden_features) / hidden_omega_0,
            )

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        h_feat = self.net_feat(coords)
        h_class = self.net_class(coords)
        h_thresh = self.net_thresh(coords)

        out_features = self.head_feat(h_feat)
        out_classes = self.head_class(h_class)
        out_thresholds = self.head_thresh(h_thresh)

        out_features = F.softmax(out_features, dim=-1)
        out_classes = F.softmax(out_classes, dim=-1)

        return out_features, out_classes, out_thresholds, coords

