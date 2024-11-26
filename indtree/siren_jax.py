import jax
import jax.numpy as jnp
from flax import linen as nn


DICT_MULTIOUTPUT_MODELS = {}


def register_model(name):
    def decorator(cls):
        DICT_MULTIOUTPUT_MODELS[name] = cls
        return cls

    return decorator


class SineLayer(nn.Module):
    in_features: int
    out_features: int
    is_first: bool = False
    omega_0: float = 30.0

    @nn.compact
    def __call__(self, x):
        # Custom initialization
        if self.is_first:
            limit = 1 / self.in_features
        else:
            limit = jnp.sqrt(6 / self.in_features) / self.omega_0

        # Define the kernel initializer
        kernel_init = nn.initializers.uniform(-limit, limit)

        # Apply Dense layer with custom kernel initializer
        x = nn.Dense(
            features=self.out_features, use_bias=True, kernel_init=kernel_init
        )(x)

        # Apply sine activation
        return jnp.sin(self.omega_0 * x)


@register_model("siren")
class Siren(nn.Module):
    in_features: int
    hidden_features: int
    hidden_layers: int
    out_features: int
    out_classes: int
    first_omega_0: float = 30.0
    hidden_omega_0: float = 30.0

    @nn.compact
    def __call__(self, coords):
        # Input layer
        h = SineLayer(
            in_features=self.in_features,
            out_features=self.hidden_features,
            is_first=True,
            omega_0=self.first_omega_0,
        )(coords)

        # Hidden layers
        for _ in range(self.hidden_layers - 1):
            h = SineLayer(
                in_features=self.hidden_features,
                out_features=self.hidden_features,
                is_first=False,
                omega_0=self.hidden_omega_0,
            )(h)

        # Output heads
        out_features = nn.Dense(self.out_features)(h)
        out_classes = nn.Dense(self.out_classes)(h)

        # Custom initialization for head3
        limit = jnp.sqrt(6 / self.hidden_features) / self.hidden_omega_0
        kernel_init = nn.initializers.uniform(-limit, limit)
        out_thresholds = nn.Dense(features=1, kernel_init=kernel_init)(h)

        # Apply softmax
        out_features = nn.softmax(out_features, axis=-1)
        out_classes = nn.softmax(out_classes, axis=-1)

        return out_features, out_classes, out_thresholds, coords


@register_model("siren_single_branch")
class SirenSingleBranch(nn.Module):
    in_features: int
    hidden_features: int
    hidden_layers: int
    out_features: int
    out_classes: int
    first_omega_0: float = 30.0
    hidden_omega_0: float = 30.0

    def setup(self):
        # Build branches
        self.net_feat = self.build_branch("feat")
        self.net_class = self.build_branch("class")
        self.net_thresh = self.build_branch("thresh")

        # Heads with custom initialization for head_thresh
        self.head_feat = nn.Dense(self.out_features)
        self.head_class = nn.Dense(self.out_classes)
        init_range = jnp.sqrt(6 / self.hidden_features) / self.hidden_omega_0
        weight_init = nn.initializers.uniform(-init_range, init_range)
        self.head_thresh = nn.Dense(1, kernel_init=weight_init)

    def build_branch(self, branch_name):
        layers = [
            SineLayer(
                in_features=self.in_features,
                out_features=self.hidden_features,
                is_first=True,
                omega_0=self.first_omega_0,
            )
        ]
        for _ in range(self.hidden_layers):
            layers.append(
                SineLayer(
                    in_features=self.hidden_features,
                    out_features=self.hidden_features,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )
        return nn.Sequential(layers)

    def __call__(self, coords):
        h_feat = self.net_feat(coords)
        h_class = self.net_class(coords)
        h_thresh = self.net_thresh(coords)

        out_features = self.head_feat(h_feat)
        out_classes = self.head_class(h_class)
        out_thresholds = self.head_thresh(h_thresh)

        out_features = nn.softmax(out_features, axis=-1)
        out_classes = nn.softmax(out_classes, axis=-1)

        return out_features, out_classes, out_thresholds, coords
