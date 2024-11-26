import copy

from indtree.direct_encoding_utils import (
    jakowski_to_direct_embedding,
    direct_embedding_to_jakowski,
)
from sklearn.ensemble import RandomForestClassifier
from indtree.gentree_tree_encoder import JakowskiEncoder
from indtree.utils import get_max_n_nodes
import numpy as np
from indtree.direct_encoding_utils import direct_embedding_to_repr3rows_batched


def generate_train_val_test_datasets_from_separate_rf(
    X,
    y,
    encodings="direct",
    n_train=200,
    n_val=100,
    n_test=100,
    random_state=42,
    max_depth=None,
    **rf_params,
):
    if encodings == "direct":
        # FIXME: there was a problem with max_depth, i.e., different random forests can have different max_depths.
        #  Now it should be ok at least when max_depth is not None
        encoding_fun = random_forest_to_direct_encodings
    elif encodings == "repr3rows":
        encoding_fun = random_forest_to_repr3rows
    else:
        raise ValueError(f"Unknown encodings: {encodings}")
    (
        feats_emb_train,
        classes_emb_train,
        thresholds_emb_train,
        rf_train,
    ) = encoding_fun(
        X,
        y,
        n_estimators=n_train,
        max_depth=max_depth,
        random_state=random_state,
        **rf_params,
    )
    (feats_emb_val, classes_emb_val, thresholds_emb_val, rf_val,) = encoding_fun(
        X,
        y,
        n_estimators=n_val,
        max_depth=max_depth,
        random_state=random_state + 1,
        **rf_params,
    )
    (feats_emb_test, classes_emb_test, thresholds_emb_test, rf_test,) = encoding_fun(
        X,
        y,
        n_estimators=n_test,
        max_depth=max_depth,
        random_state=random_state + 2,
        **rf_params,
    )
    return (
        feats_emb_train,
        classes_emb_train,
        thresholds_emb_train,
        rf_train,
        feats_emb_val,
        classes_emb_val,
        thresholds_emb_val,
        rf_val,
        feats_emb_test,
        classes_emb_test,
        thresholds_emb_test,
        rf_test,
    )


def generate_tree_dataset(X, y, max_depth=None, **rf_params):
    trees, max_depth = generate_tree_classifiers_with_random_forest(
        X, y, max_depth, **rf_params
    )
    n_features_in = X.shape[1]
    n_classes = len(np.unique(y))
    (
        feats_embeddings,
        classes_embeddings,
        thresholds_embeddings,
    ) = trees_to_direct_encodings(trees, max_depth, n_features_in, n_classes)
    return feats_embeddings, classes_embeddings, thresholds_embeddings


def generate_tree_classifiers_with_random_forest(X, y, max_depth=None, **rf_params):
    rf = RandomForestClassifier(max_depth=max_depth, **rf_params)
    rf.fit(X, y)
    max_depth = np.max([estimator.get_depth() for estimator in rf.estimators_])
    return rf.estimators_, max_depth


def random_forest_to_direct_encodings(X, y, max_depth=None, **rf_params):
    rf = RandomForestClassifier(max_depth=max_depth, **rf_params)
    rf.fit(X, y)
    if max_depth is None:
        max_depth = np.max([estimator.get_depth() for estimator in rf.estimators_])
    n_features_in = X.shape[1]
    n_classes = len(np.unique(y))
    (
        feats_embeddings,
        classes_embeddings,
        thresholds_embeddings,
    ) = trees_to_direct_encodings(rf.estimators_, max_depth, n_features_in, n_classes)
    return feats_embeddings, classes_embeddings, thresholds_embeddings, rf


def random_forest_to_repr3rows(X, y, max_depth=None, **rf_params):
    (
        feats_embeddings,
        classes_embeddings,
        thresholds_embeddings,
        rf,
    ) = random_forest_to_direct_encodings(X, y, max_depth, **rf_params)
    # print(f'random_forest_to_repr3rows) feat_emb shape: {feats_embeddings.shape}, class_emb shape:
    # {classes_embeddings.shape}, threshold_emb shape: {thresholds_embeddings.shape}')
    (
        features_nodes_class_all,
        leaf_nodes_class_all,
        threshold_all,
    ) = direct_embedding_to_repr3rows_batched(
        feats_embeddings, classes_embeddings, thresholds_embeddings
    )
    return features_nodes_class_all, leaf_nodes_class_all, threshold_all, rf


def direct_encodings_to_random_forest(
    feats_embeddings, classes_embeddings, thresholds_embeddings, original_rf
):
    trees = direct_encodings_to_trees(
        feats_embeddings, classes_embeddings, thresholds_embeddings
    )
    rf_copy = copy.deepcopy(original_rf)
    rf_copy.estimators_ = trees
    return rf_copy


def trees_to_direct_encodings(trees, max_depth, n_features_in, n_classes):
    encoder = JakowskiEncoder(n_features=n_features_in, n_classes=n_classes)
    n_nodes = get_max_n_nodes(max_depth)
    feats_embeddings = np.full((len(trees), n_nodes, n_features_in), np.nan)
    classes_embeddings = np.full((len(trees), n_nodes, n_classes), np.nan)
    thresholds_embeddings = np.full((len(trees), n_nodes, 1), np.nan)
    for i, tree in enumerate(trees):
        enc = encoder.encode(tree, depth=max_depth)
        feats, classes, thresholds = jakowski_to_direct_embedding(
            enc, n_features_in, n_classes
        )
        feats_embeddings[i] = feats
        classes_embeddings[i] = classes
        thresholds_embeddings[i] = thresholds
    return feats_embeddings, classes_embeddings, thresholds_embeddings


def trees_to_repr3rows(trees, max_depth, n_features_in, n_classes):
    (
        feats_embeddings,
        classes_embeddings,
        thresholds_embeddings,
    ) = trees_to_direct_encodings(trees, max_depth, n_features_in, n_classes)
    return direct_embeddings_to_repr3rows(
        feats_embeddings, classes_embeddings, thresholds_embeddings
    )


def direct_encodings_to_trees(
    feats_embeddings, classes_embeddings, thresholds_embeddings
):
    n_features_in = feats_embeddings.shape[-1]
    n_classes = classes_embeddings.shape[-1]
    encoder = JakowskiEncoder(n_features=n_features_in, n_classes=n_classes)
    n_trees = feats_embeddings.shape[0]
    trees = list()
    for i in range(n_trees):
        enc = direct_embedding_to_jakowski(
            features_onehot=feats_embeddings[i],
            class_onehot=classes_embeddings[i],
            threshold=thresholds_embeddings[i],
        )
        tree = encoder.decode(enc)
        trees.append(tree)
    return trees
