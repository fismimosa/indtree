import numpy as np


def jakowski_to_direct_embeddings(jakowski_encoding, n_features, n_classes):
    return np.array(
        [
            jakowski_to_direct_embedding(jakowski_encoding[i], n_features, n_classes)
            for i in jakowski_encoding.shape[0]
        ]
    )


def jakowski_to_direct_embedding(jakowski_encoding, n_features, n_classes):
    features_onehot = jakowski_features_to_features_onehot(
        jakowski_encoding[0], n_features
    )
    class_onehot, threshold = jakowski_threshold_to_threshold_and_class_onehot(
        jakowski_encoding[1], n_classes
    )
    return features_onehot, class_onehot, threshold


def direct_embeddings_to_jakowski(features_onehot, class_onehot, threshold):
    return np.array(
        [
            direct_embedding_to_jakowski(
                features_onehot[i], class_onehot[i], threshold[i]
            )
            for i in range(len(features_onehot))
        ]
    )


def direct_embedding_to_jakowski(features_onehot, class_onehot, threshold):
    features = features_onehot_to_jakowski(features_onehot).reshape(1, -1)
    threshold_and_class = threshold_and_class_onehot_to_jakowski(
        class_onehot, threshold
    ).reshape(1, -1)
    return np.concatenate([features, threshold_and_class], axis=0)


def jakowski_features_to_features_onehot(features, n_features):
    max_depth = int(np.log2(features.shape[0] + 1) - 1)
    internal_nodes_onehot = np.eye(n_features)[
        features[: 2**max_depth - 1].astype(int) - 1
    ]
    leaf_nodes_onehot = np.zeros((2**max_depth, n_features))
    return np.concatenate([internal_nodes_onehot, leaf_nodes_onehot], axis=0)


def jakowski_threshold_to_threshold_and_class_onehot(threshold_and_class, n_classes):
    max_depth = int(np.log2(threshold_and_class.shape[0] + 1) - 1)
    leaf_nodes_class_onehot = np.eye(n_classes)[
        threshold_and_class[2**max_depth - 1 :].astype(int) - 1
    ]
    internal_nodes_threshold_onehot = np.zeros((2**max_depth - 1, n_classes))
    class_onehot = np.concatenate(
        [internal_nodes_threshold_onehot, leaf_nodes_class_onehot], axis=0
    )

    internal_nodes_threshold = threshold_and_class[: 2**max_depth - 1].reshape(-1, 1)
    leaf_nodes_threshold = np.full((2**max_depth, 1), -1).reshape(
        -1, 1
    )  # -1 is a placeholder for leaf nodes
    threshold = np.concatenate([internal_nodes_threshold, leaf_nodes_threshold], axis=0)

    return class_onehot, threshold


def features_onehot_to_jakowski(features_onehot):
    max_depth = int(np.log2(features_onehot.shape[0] + 1) - 1)
    internal_nodes_features = (
        np.argmax(features_onehot[: 2**max_depth - 1], axis=1) + 1
    )
    leaf_nodes_features = np.full(
        2**max_depth, -1
    )  # -1 is a placeholder for leaf nodes
    return np.concatenate([internal_nodes_features, leaf_nodes_features], axis=0)


def threshold_and_class_onehot_to_jakowski(class_onehot, threshold):
    max_depth = int(np.log2(class_onehot.shape[0] + 1) - 1)
    leaf_nodes_class = np.argmax(class_onehot[2**max_depth - 1 :], axis=1) + 1
    internal_nodes_threshold = threshold[: 2**max_depth - 1].ravel()
    return np.concatenate([internal_nodes_threshold, leaf_nodes_class], axis=0)


def direct_embedding_to_repr3rows(features_onehot, class_onehot):
    leaf_nodes_class = class_onehot.argmax(axis=1).reshape(-1, 1)
    features_nodes_class = features_onehot.argmax(axis=1).reshape(-1, 1)
    # return np.concatenate([features_nodes_class, leaf_nodes_class, threshold], axis=1)
    return features_nodes_class, leaf_nodes_class


def direct_embedding_to_repr3rows_batched(features_onehot, class_onehot, threshold):
    assert len(features_onehot) == len(class_onehot) == len(threshold)
    features_nodes_class_all = np.full(
        (features_onehot.shape[0], features_onehot.shape[1], 1), np.nan
    )
    leaf_nodes_class_all = np.full(
        (class_onehot.shape[0], class_onehot.shape[1], 1), np.nan
    )
    for i in range(features_onehot.shape[0]):
        features_nodes, leaf_nodes = direct_embedding_to_repr3rows(
            features_onehot[i], class_onehot[i]
        )
        features_nodes_class_all[i] = features_nodes
        leaf_nodes_class_all[i] = leaf_nodes

    return features_nodes_class_all, leaf_nodes_class_all, threshold


def repr3rows_to_direct_embedding(features_nodes, leaf_nodes, n_features, n_classes):
    # print(f'features_nodes shape: {features_nodes.shape}, leaf_nodes shape: {leaf_nodes.shape}')
    # print('features_nodes', features_nodes)
    # print('leaf_nodes', leaf_nodes)
    features_onehot = np.eye(n_features)[
        np.ceil(features_nodes.ravel() - 0.5).astype(int)
    ]
    leaf_nodes_class = np.eye(n_classes)[np.ceil(leaf_nodes.ravel() - 0.5).astype(int)]
    return features_onehot, leaf_nodes_class


# TODO for train_hydra_esann_mse val
def repr3rows_to_direct_embedding_batched(
    features_nodes, leaf_nodes, threshold, n_features, n_classes
):
    assert len(features_nodes) == len(leaf_nodes) == len(threshold)

    features_onehot_all = np.full(
        (features_nodes.shape[0], features_nodes.shape[1], n_features), np.nan
    )
    leaf_nodes_class_all = np.full(
        (leaf_nodes.shape[0], leaf_nodes.shape[1], n_classes), np.nan
    )
    for i in range(features_nodes.shape[0]):
        features_onehot, leaf_nodes_class = repr3rows_to_direct_embedding(
            features_nodes[i], leaf_nodes[i], n_features, n_classes
        )
        features_onehot_all[i] = features_onehot
        leaf_nodes_class_all[i] = leaf_nodes_class

    return features_onehot_all, leaf_nodes_class_all, threshold
