from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED
import numpy as np
import copy
from indtree.utils import get_project_root


DATASETS = [
    "iris",
    "car",
    "pima",
    "australian",
    "lymph",
    "breast",
    "heart",
    "ecoli",
    "led7",
    "bank",
    "adult",
    "compas",
    "german",
    "churn",
    "sonar",
    "fico",
    "drybean",
    "avila",
    "banknote",
    "isolet",
    "wine",
    "yeast",
    "glass",
    "vehicle",
    "egg",
]


# LOAD A DATASET
def load_data(dataset, path=str(get_project_root() / "../datasets/") + "/"):

    X_train_path = path + dataset + "/preprocessed/X_train.csv"
    X_val_path = path + dataset + "/preprocessed/X_val.csv"
    X_test_path = path + dataset + "/preprocessed/X_test.csv"
    y_train_path = path + dataset + "/preprocessed/y_train.csv"
    y_val_path = path + dataset + "/preprocessed/y_val.csv"
    y_test_path = path + dataset + "/preprocessed/y_test.csv"

    X_train = np.loadtxt(X_train_path, delimiter=",")
    X_val = np.loadtxt(X_val_path, delimiter=",")
    X_test = np.loadtxt(X_test_path, delimiter=",")
    y_train = np.loadtxt(y_train_path, delimiter=",")
    y_val = np.loadtxt(y_val_path, delimiter=",")
    y_test = np.loadtxt(y_test_path, delimiter=",")

    # Check dimensions
    if X_train.shape[1] != X_test.shape[1]:
        print("ERROR: X_train.shape[1] != X_test.shape[1]")
        return

    if X_train.shape[1] != X_val.shape[1]:
        print("ERROR: X_train.shape[1] != X_val.shape[1]")
        return

    if X_train.shape[0] != y_train.shape[0]:
        print("ERROR: X_train.shape[0] != y_train.shape[0]")
        return

    if X_val.shape[0] != y_val.shape[0]:
        print("ERROR: X_val.shape[0] != y_val.shape[0]")
        return

    if X_test.shape[0] != y_test.shape[0]:
        print("ERROR: X_test.shape[0] != y_test.shape[0]")
        return

    return [X_train, X_val, X_test, y_train, y_val, y_test]


# PRUNE DUPLICATE LEAVES
def refine_prune_index(clf):
    # Corrects the prune index function by changing the state of the decision tree
    children_left = [x for x in clf.tree_.children_left if x != -1]
    children_right = [x for x in clf.tree_.children_right if x != -1]

    nodes = children_left + children_right
    nodes.append(0)  # root
    nodes.sort()

    n_nodes = len(nodes)  # number of nodes in the tree

    state = clf.tree_.__getstate__()

    state["node_count"] = n_nodes
    state["max_depth"] = get_depth(clf.tree_)
    new_nodes = np.zeros(
        (n_nodes,),
        dtype=[
            ("left_child", "<i8"),
            ("right_child", "<i8"),
            ("feature", "<i8"),
            ("threshold", "<f8"),
            ("impurity", "<f8"),
            ("n_node_samples", "<i8"),
            ("weighted_n_node_samples", "<f8"),
            ("missing_go_to_left", "u1"),
        ],
    )
    new_values = np.zeros((n_nodes, clf.n_outputs_, clf.n_classes_))

    i = 0  # index in the new state
    index = {}  # {node index in the old state : node index in the new state}
    index[0] = 0  # root
    parent = {}  # {node index in the old state : parent index in the new state}
    for node in nodes:
        index[node] = i
        new_nodes[index[node]] = state["nodes"][node]
        new_values[index[node]] = state["values"][node]

        if state["nodes"][node][0] != -1:  # not a leaf
            parent[state["nodes"][node][0]] = (index[node], 0)  # left child
            parent[state["nodes"][node][1]] = (index[node], 1)  # right child

        if node != 0:  # non root node
            new_nodes[parent[node][0]][parent[node][1]] = index[
                node
            ]  # update the parent

        i += 1

    state["nodes"] = copy.deepcopy(new_nodes)
    state["values"] = copy.deepcopy(new_values)

    clf.tree_.__setstate__(state)


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (
        inner_tree.children_left[index] == TREE_LEAF
        and inner_tree.children_right[index] == TREE_LEAF
    )


def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    children_left = len([x for x in inner_tree.children_left if x != -1])
    children_right = len([x for x in inner_tree.children_right if x != -1])

    n_nodes = children_left + children_right + 1

    if n_nodes > 3:
        # Prune children if both children are leaves now and make the same decision:
        if (
            is_leaf(inner_tree, inner_tree.children_left[index])
            and is_leaf(inner_tree, inner_tree.children_right[index])
            and (decisions[index] == decisions[inner_tree.children_left[index]])
            and (decisions[index] == decisions[inner_tree.children_right[index]])
        ):
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
            inner_tree.threshold[index] = TREE_UNDEFINED
            inner_tree.feature[index] = TREE_UNDEFINED
            # print("Pruned {}".format(index))


def prune_duplicate_leaves(clf):
    if clf.tree_.node_count > 3:
        # Remove leaves if both
        decisions = (
            clf.tree_.value.argmax(axis=2).flatten().tolist()
        )  # Decision for each node
        prune_index(clf.tree_, decisions)
        refine_prune_index(clf)


def leaves_count(clf):
    # Returns the number of leaves of the tree.

    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right

    n_leaves = 0

    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        node_id, depth = stack.pop()

        # If the left and right child of a node is not the same we have a split
        is_split_node = children_left[node_id] != children_right[node_id]

        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            n_leaves += 1

    return n_leaves


def max_node_count(max_depth):
    # Returns the maximum number of nodes a tree can have, given its maximum depth
    nodes = 0
    for i in range(0, max_depth + 1):
        nodes += pow(2, i)
    return nodes


def get_depth(inner_tree, node=0, depth=0, verbose=False):
    if verbose:
        print("Node: {} - depth: {}".format(node, depth))

    if (
        inner_tree.children_left[node] != TREE_LEAF
        and inner_tree.children_right[node] != TREE_LEAF
    ):
        return max(
            get_depth(
                inner_tree, inner_tree.children_left[node], depth + 1, verbose=verbose
            ),
            get_depth(
                inner_tree, inner_tree.children_right[node], depth + 1, verbose=verbose
            ),
        )

    return depth
