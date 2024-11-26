import jax.numpy as jnp
import jax


def integer_bfs_index(n):
    return jnp.arange(n).reshape(-1, 1)


def binary_bfs_index(n, offset=1, max_len=None):
    if max_len is None:
        max_len = (n + offset - 1).bit_length()
    numbers = jnp.arange(offset, n + offset)
    bits = ((numbers[:, None] & (1 << jnp.arange(max_len)))) > 0
    bits = bits.astype(int)
    bits = bits[:, ::-1]  # Reverse bits to match binary string order
    return bits


class TreeFittingSplit:
    def __init__(self, feat, label, thresh, coords_type: str = "integer"):
        self.feat = jnp.array(feat)
        self.label = jnp.array(label)
        self.thresh = jnp.array(thresh)
        if coords_type == "integer":
            self.coords = integer_bfs_index(self.feat.shape[0])
        elif coords_type == "binary":
            self.coords = binary_bfs_index(self.feat.shape[0])
        else:
            raise ValueError(
                f"coords_type must be either 'integer' or 'binary', got {coords_type}"
            )
        assert len(self.feat) == len(self.label) == len(self.thresh), (
            f"Lengths of onehot_feat, onehot_class, and onehot_thresh must be equal, "
            f"got {len(self.feat)}, {len(self.label)}, {len(self.thresh)}"
        )
        self.length = self.label.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            self.coords[idx],
            self.feat[idx],
            self.label[idx],
            self.thresh[idx],
        )

    def coords_len(self):
        return self.coords.shape[1]


# Example script to test the above class and functions
def main():
    # Sample data
    num_samples = 8  # Number of samples
    num_features = 4  # Number of features for one-hot encoding
    num_classes = 3  # Number of classes for one-hot encoding

    # Create sample one-hot encoded features
    onehot_feat = jax.nn.one_hot(jnp.arange(num_samples) % num_features, num_features)
    # Create sample one-hot encoded classes
    onehot_class = jax.nn.one_hot(jnp.arange(num_samples) % num_classes, num_classes)
    # Create sample thresholds
    onehot_thresh = jnp.linspace(0, 1, num_samples).reshape(-1, 1)

    print("Testing with integer coordinates:")
    dataset_integer = TreeFittingSplit(
        onehot_feat, onehot_class, onehot_thresh, coords_type="integer"
    )
    for idx in range(len(dataset_integer)):
        coords, feat, class_, thresh = dataset_integer[idx]
        print(f"Index {idx}:")
        print(f"  Coords: {coords}")
        print(f"  Feature: {feat}")
        print(f"  Class: {class_}")
        print(f"  Threshold: {thresh}\n")

    print("Testing with binary coordinates:")
    dataset_binary = TreeFittingSplit(
        onehot_feat, onehot_class, onehot_thresh, coords_type="binary"
    )
    for idx in range(len(dataset_binary)):
        coords, feat, class_, thresh = dataset_binary[idx]
        print(f"Index {idx}:")
        print(f"  Coords: {coords}")
        print(f"  Feature: {feat}")
        print(f"  Class: {class_}")
        print(f"  Threshold: {thresh}\n")

    # Test integer_bfs_index function separately
    print("Testing integer_bfs_index function:")
    int_coords = integer_bfs_index(num_samples)
    print(int_coords)

    # Test binary_bfs_index function separately
    print("Testing binary_bfs_index function:")
    bin_coords = binary_bfs_index(num_samples)
    print(bin_coords)


if __name__ == "__main__":
    main()
