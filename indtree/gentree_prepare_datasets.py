import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff
from sklearn.pipeline import make_pipeline

encoders = {
    "label": LabelEncoder(),
    "ordinal": make_pipeline(OrdinalEncoder(), StandardScaler()),
    "nominal": make_pipeline(
        OneHotEncoder(sparse_output=False, handle_unknown="ignore"), StandardScaler()
    ),
    "numerical": StandardScaler(),
}

data = {"label": "y", "ordinal": "X", "nominal": "X"}


def data_encoder(
    train, val, test, path, encoder
):  # numpy.array, numpy.array, string, string, bool

    enc = encoders[encoder]
    enc.fit(train)

    if encoder == "ordinal" or encoder == "nominal":
        print("Categories: {}".format(enc[0].categories_))
    elif encoder == "label":
        print("Classes: {}".format(enc.classes_))

    train = enc.transform(train)
    val = enc.transform(val)
    test = enc.transform(test)

    return [train, val, test]


def split_train_val_test(
    df, path, test_size=0.20, random_state=1, verbose=False
):  # dataframe, string
    data = df.values

    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=random_state
    )

    pd.DataFrame(X_train).to_csv(path + "/X_train.csv", index=False, header=False)
    pd.DataFrame(y_train).to_csv(path + "/y_train.csv", index=False, header=False)
    pd.DataFrame(X_val).to_csv(path + "/X_val.csv", index=False, header=False)
    pd.DataFrame(y_val).to_csv(path + "/y_val.csv", index=False, header=False)
    pd.DataFrame(X_test).to_csv(path + "/X_test.csv", index=False, header=False)
    pd.DataFrame(y_test).to_csv(path + "/y_test.csv", index=False, header=False)

    if verbose:
        print("Train X", X_train.shape)
        print("Train y", y_train.shape)
        print("Val X", X_val.shape)
        print("Val y", y_val.shape)
        print("Test X", X_test.shape)
        print("Test y", y_test.shape)

    return [X_train, X_val, X_test, y_train, y_val, y_test]


def save_dataset(path, X_train, X_val, X_test, y_train, y_val, y_test):
    pd.DataFrame(X_train).to_csv(
        path + "/preprocessed/X_train.csv", index=False, header=False
    )
    pd.DataFrame(X_test).to_csv(
        path + "/preprocessed/X_test.csv", index=False, header=False
    )
    pd.DataFrame(X_val).to_csv(
        path + "/preprocessed/X_val.csv", index=False, header=False
    )

    pd.DataFrame(y_train).to_csv(
        path + "/preprocessed/y_train.csv", index=False, header=False
    )
    pd.DataFrame(y_test).to_csv(
        path + "/preprocessed/y_test.csv", index=False, header=False
    )
    pd.DataFrame(y_val).to_csv(
        path + "/preprocessed/y_val.csv", index=False, header=False
    )


def prepare_iris_dataset(path, filename, test_size=0.20):

    column_names = [
        "sepal length",
        "sepal width",
        "petal length",
        "petal width",
        "class",
    ]

    # Read Dataset
    df = pd.read_csv(path + filename, names=column_names)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".data", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_car_dataset(path, filename, test_size=0.20):
    column_names = [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug_boot",
        "safety",
        "class",
    ]

    # Read Dataset
    df = pd.read_csv(path + filename, names=column_names)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".data", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="ordinal"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_pima_dataset(path, filename, test_size=0.20):

    # Read Dataset
    df = pd.read_csv(path + filename)
    df = df.rename(columns={"Outcome": "class"})

    column_names = list(df.columns)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace("-diabetes.csv", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_australian_dataset(path, filename, test_size=0.20):

    column_names = [str(item) for item in range(0, 14)]
    column_names.append("class")

    # Read Dataset
    df = pd.read_csv(path + filename, names=column_names, sep=" ")

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".dat", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Binary
    for col in ["0", "7", "8", "10"]:
        cat_col = df.pop(col)
        df.insert(14, col, cat_col)

    class_col = df.pop("class")
    df.insert(14, "class", class_col)

    # Numerical
    for col in ["1", "2", "6", "9", "12", "13"]:
        cat_col = df.pop(col)
        df.insert(14, col, cat_col)

    class_col = df.pop("class")
    df.insert(14, "class", class_col)

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    X_train_cat = X_train[:, :4]
    X_val_cat = X_val[:, :4]
    X_test_cat = X_test[:, :4]

    X_train_bin = X_train[:, 4:8]
    X_val_bin = X_val[:, 4:8]
    X_test_bin = X_test[:, 4:8]

    X_train_num = X_train[:, 8:]
    X_val_num = X_val[:, 8:]
    X_test_num = X_test[:, 8:]

    # Attributes encoder
    X_train_num, X_val_num, X_test_num = data_encoder(
        train=X_train_num,
        val=X_val_num,
        test=X_test_num,
        path=path,
        encoder="numerical",
    )
    X_train_cat, X_val_cat, X_test_cat = data_encoder(
        train=X_train_cat, val=X_val_cat, test=X_test_cat, path=path, encoder="nominal"
    )

    X_train = np.concatenate((X_train_cat, X_train_num), axis=1)
    X_train = np.concatenate((X_train, X_train_bin), axis=1)

    X_val = np.concatenate((X_val_cat, X_val_num), axis=1)
    X_val = np.concatenate((X_val, X_val_bin), axis=1)

    X_test = np.concatenate((X_test_cat, X_test_num), axis=1)
    X_test = np.concatenate((X_test, X_test_bin), axis=1)

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_lymph_dataset(path, filename, test_size=0.20):

    # Read Dataset
    df = pd.read_csv(path + filename)
    df = df.rename(columns={"Y": "class"})

    column_names = list(df.columns)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    df = df[df["class"] != "y1"]
    df = df[df["class"] != "y4"]
    df["class"] = df["class"].map({"y2": 0, "y3": 1}).astype(int)

    # Save it in csv
    name = filename.replace(".txt", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Binary
    mapping = {1: 0, 2: 1}
    bin_cols = ["X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X16", "X17"]

    for col in bin_cols:
        df[col] = df[col].map(mapping).astype(int)
        bin_col = df.pop(col)
        df.insert(18, col, bin_col)

    class_col = df.pop("class")
    df.insert(18, "class", class_col)

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Categorical
    X_train_cat = X_train[:, :8]
    X_val_cat = X_val[:, :8]
    X_test_cat = X_test[:, :8]

    # Binary
    X_train_bin = X_train[:, 8:]
    X_val_bin = X_val[:, 8:]
    X_test_bin = X_test[:, 8:]

    # Attributes encoder
    X_train_cat, X_val_cat, X_test_cat = data_encoder(
        train=X_train_cat, val=X_val_cat, test=X_test_cat, path=path, encoder="nominal"
    )

    X_train = np.concatenate((X_train_cat, X_train_bin), axis=1)
    X_val = np.concatenate((X_val_cat, X_val_bin), axis=1)
    X_test = np.concatenate((X_test_cat, X_test_bin), axis=1)

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_breast_dataset(path, filename, test_size=0.20):

    column_names = [str(item) for item in range(0, 10)]
    column_names.append("class")

    # Read Dataset
    df = pd.read_csv(path + filename, names=column_names)

    # Remove id
    df = df.drop("0", axis=1)

    df["6"] = df["6"].replace("?", int(np.mean(df["6"][df["6"] != "?"].map(int))))

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".data", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_heart_dataset(path, filename, test_size=0.30):

    column_names = [str(x) for x in range(1, 14)]
    column_names.append("class")

    # Read Dataset
    df = pd.read_csv(path + filename, names=column_names, sep=" ")

    mapping = {1.0: 1, 0.0: 0}
    df["2"] = df["2"].map(mapping).astype(int)
    df["6"] = df["6"].map(mapping).astype(int)
    df["9"] = df["9"].map(mapping).astype(int)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace("-statlog.dat", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Binary
    for col in ["2", "6", "9"]:
        bin_col = df.pop(col)
        df.insert(13, col, bin_col)

    class_col = df.pop("class")
    df.insert(13, "class", class_col)

    # Categorical (ordinal)
    for col in ["11"]:
        cat_col = df.pop(col)
        df.insert(13, col, cat_col)

    class_col = df.pop("class")
    df.insert(13, "class", class_col)

    # Numerical
    for col in ["1", "4", "5", "8", "10", "12"]:
        num_col = df.pop(col)
        df.insert(13, col, num_col)

    class_col = df.pop("class")
    df.insert(13, "class", class_col)

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    X_train_cat_nom = X_train[:, :3]
    X_val_cat_nom = X_val[:, :3]
    X_test_cat_nom = X_test[:, :3]

    X_train_bin = X_train[:, 3:6]
    X_val_bin = X_val[:, 3:6]
    X_test_bin = X_test[:, 3:6]

    X_train_cat_ord = X_train[:, 6:7]
    X_val_cat_ord = X_val[:, 6:7]
    X_test_cat_ord = X_test[:, 6:7]

    X_train_num = X_train[:, 7:]
    X_val_num = X_val[:, 7:]
    X_test_num = X_test[:, 7:]

    # Attributes encoder (nominal)
    X_train_cat_nom, X_val_cat_nom, X_test_cat_nom = data_encoder(
        train=X_train_cat_nom,
        val=X_val_cat_nom,
        test=X_test_cat_nom,
        path=path,
        encoder="nominal",
    )

    # Attributes encoder (ordinal)
    X_train_cat_ord, X_val_cat_ord, X_test_cat_ord = data_encoder(
        train=X_train_cat_ord,
        val=X_val_cat_ord,
        test=X_test_cat_ord,
        path=path,
        encoder="ordinal",
    )

    # Attributes encoder (numerical)
    X_train_num, X_val_num, X_test_num = data_encoder(
        train=X_train_num,
        val=X_val_num,
        test=X_test_num,
        path=path,
        encoder="numerical",
    )

    X_train_cat = np.concatenate((X_train_cat_nom, X_train_cat_ord), axis=1)
    X_train_cat = np.concatenate((X_train_cat, X_train_bin), axis=1)
    X_train = np.concatenate((X_train_cat, X_train_num), axis=1)

    X_val_cat = np.concatenate((X_val_cat_nom, X_val_cat_ord), axis=1)
    X_val_cat = np.concatenate((X_val_cat, X_val_bin), axis=1)
    X_val = np.concatenate((X_val_cat, X_val_num), axis=1)

    X_test_cat = np.concatenate((X_test_cat_nom, X_test_cat_ord), axis=1)
    X_test_cat = np.concatenate((X_test_cat, X_test_bin), axis=1)
    X_test = np.concatenate((X_test_cat, X_test_num), axis=1)

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_ecoli_dataset(path, filename, test_size=0.30):

    column_names = [str(x) for x in range(1, 9)]
    column_names.append("class")

    # Read Dataset
    df = pd.read_csv(path + filename, names=column_names, sep="  ", engine="python")

    # Remove name
    df = df.drop("1", axis=1)
    df = df.drop("5", axis=1)

    mapping_class = {
        " cp": 0,
        " im": 1,
        "pp": 2,
        "imU": 3,
        "om": 4,
        "omL": 5,
        "imL": 6,
        "imS": 7,
    }
    df["class"] = df["class"].map(mapping_class).astype(int)
    df = df[df["class"] < 5]

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".data", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    X_train_num = X_train[:, :5]
    X_val_num = X_val[:, :5]
    X_test_num = X_test[:, :5]

    X_train_cat = X_train[:, 5:]
    X_val_cat = X_val[:, 5:]
    X_test_cat = X_test[:, 5:]

    # Attributes encoder
    X_train_num, X_val_num, X_test_num = data_encoder(
        train=X_train_num,
        val=X_val_num,
        test=X_test_num,
        path=path,
        encoder="numerical",
    )

    X_train = np.concatenate((X_train_cat, X_train_num), axis=1)
    X_val = np.concatenate((X_val_cat, X_val_num), axis=1)
    X_test = np.concatenate((X_test_cat, X_test_num), axis=1)

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_led7_dataset(path, filename, test_size=0.30):

    # Read Dataset
    df = pd.read_csv(path + filename, sep=",", engine="python")
    df = df.rename(columns={"Y": "class"})

    column_names = list(df.columns)

    df["class"] = (
        df["class"]
        .map(
            {
                "y0": 0,
                "y1": 1,
                "y2": 2,
                "y3": 3,
                "y4": 4,
                "y5": 5,
                "y6": 6,
                "y7": 7,
                "y8": 8,
                "y9": 9,
            }
        )
        .astype(int)
    )
    df = df[df["class"] < 8]

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".txt", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_bank_dataset(path, filename, test_size=0.20):

    # Read Dataset
    df = pd.read_csv(path + filename, sep=",", engine="python")
    df = df.rename(columns={"y": "class"})

    column_names = list(df.columns)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".txt", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    mapping_class = {"no": 0, "yes": 1}
    for col in ["default", "housing", "loan"]:
        df[col] = df[col].map(mapping_class)

    # Categorical
    for col in ["job", "marital", "education", "contact", "month", "poutcome"]:
        cat_col = df.pop(col)
        df.insert(16, col, cat_col)

    class_col = df.pop("class")
    df.insert(16, "class", class_col)

    # Numerical
    for col in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
        num_col = df.pop(col)
        df.insert(16, col, num_col)

    class_col = df.pop("class")
    df.insert(16, "class", class_col)

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    X_train_bin = X_train[:, :3]
    X_val_bin = X_val[:, :3]
    X_test_bin = X_test[:, :3]

    X_train_cat = X_train[:, 3:9]
    X_val_cat = X_val[:, 3:9]
    X_test_cat = X_test[:, 3:9]

    X_train_num = X_train[:, 9:]
    X_val_num = X_val[:, 9:]
    X_test_num = X_test[:, 9:]

    # Attributes encoder (categorical)
    X_train_cat, X_val_cat, X_test_cat = data_encoder(
        train=X_train_cat, val=X_val_cat, test=X_test_cat, path=path, encoder="nominal"
    )

    # Attributes encoder (numerical)
    X_train_num, X_val_num, X_test_num = data_encoder(
        train=X_train_num,
        val=X_val_num,
        test=X_test_num,
        path=path,
        encoder="numerical",
    )

    X_train = np.concatenate((X_train_cat, X_train_num), axis=1)
    X_train = np.concatenate((X_train, X_train_bin), axis=1)

    X_val = np.concatenate((X_val_cat, X_val_num), axis=1)
    X_val = np.concatenate((X_val, X_val_bin), axis=1)

    X_test = np.concatenate((X_test_cat, X_test_num), axis=1)
    X_test = np.concatenate((X_test, X_test_bin), axis=1)

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_german_dataset(path, filename, test_size=0.20):

    column_names = [str(item) for item in range(0, 23)]
    column_names.append("class")

    # Read Dataset
    df = pd.read_csv(path + filename, sep="  ", names=column_names)

    # Remove missing values
    df = df.dropna()

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".data-numeric", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_compas_dataset(path, filename, test_size=0.20):

    # Read Dataset
    df = pd.read_csv(path + filename, sep=",")
    df = df.rename(columns={"y": "class"})

    column_names = list(df.columns)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".txt", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_adult_dataset(path, filename, test_size=0.20):

    # Read Dataset
    df = pd.read_csv(path + filename, sep=",")
    df = df.rename(columns={"y": "class"})

    column_names = list(df.columns)

    # Remove useless columns
    del df["fnlwgt"]
    del df["education-num"]

    # Remove Missing Values
    for col in df.columns:
        if "?" in df[col].unique():
            df[col][df[col] == "?"] = df[col].value_counts().index[0]

    mapping_class = {"Male": 0, "Female": 1}
    for col in ["sex"]:
        df[col] = df[col].map(mapping_class)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".txt", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Numerical
    for col in ["age", "capital-gain", "capital-loss", "hours-per-week"]:
        num_col = df.pop(col)
        df.insert(12, col, num_col)

    class_col = df.pop("class")
    df.insert(12, "class", class_col)

    # Binary
    for col in ["sex"]:
        num_col = df.pop(col)
        df.insert(12, col, num_col)

    class_col = df.pop("class")
    df.insert(12, "class", class_col)

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    X_train_cat = X_train[:, 0:7]
    X_val_cat = X_val[:, 0:7]
    X_test_cat = X_test[:, 0:7]

    X_train_num = X_train[:, 7:11]
    X_val_num = X_val[:, 7:11]
    X_test_num = X_test[:, 7:11]

    X_train_bin = X_train[:, 11:]
    X_val_bin = X_val[:, 11:]
    X_test_bin = X_test[:, 11:]

    # Attributes encoder (categorical)
    X_train_cat, X_val_cat, X_test_cat = data_encoder(
        train=X_train_cat, val=X_val_cat, test=X_test_cat, path=path, encoder="nominal"
    )

    # Attributes encoder (numerical)
    X_train_num, X_val_num, X_test_num = data_encoder(
        train=X_train_num,
        val=X_val_num,
        test=X_test_num,
        path=path,
        encoder="numerical",
    )

    X_train = np.concatenate((X_train_cat, X_train_num), axis=1)
    X_train = np.concatenate((X_train, X_train_bin), axis=1)

    X_val = np.concatenate((X_val_cat, X_val_num), axis=1)
    X_val = np.concatenate((X_val, X_val_bin), axis=1)

    X_test = np.concatenate((X_test_cat, X_test_num), axis=1)
    X_test = np.concatenate((X_test, X_test_bin), axis=1)

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_churn_dataset(path, filename, test_size=0.20):

    # Read Dataset
    df = pd.read_csv(path + filename, sep=",")

    df = df.drop("customerID", axis=1)
    df = df.rename(columns={"Churn": "class"})

    column_names = list(df.columns)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace("_origin.csv", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Binary
    for col in [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]:
        bin_col = df.pop(col)
        df.insert(19, col, bin_col)

    # Categorical
    for col in [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]:
        cat_col = df.pop(col)
        df.insert(19, col, cat_col)

    class_col = df.pop("class")
    df.insert(19, "class", class_col)

    df.loc[df.TotalCharges == " ", "TotalCharges"] = "0"
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    mapping_gender = {"Female": 1, "Male": 0}
    df["gender"] = df["gender"].map(mapping_gender).astype(int)

    mapping = {"Yes": 1, "No": 0}
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].map(mapping).astype(int)

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    X_train_num = X_train[:, :3]
    X_val_num = X_val[:, :3]
    X_test_num = X_test[:, :3]

    X_train_bin = X_train[:, 3:9]
    X_val_bin = X_val[:, 3:9]
    X_test_bin = X_test[:, 3:9]

    X_train_cat = X_train[:, 9:]
    X_val_cat = X_val[:, 9:]
    X_test_cat = X_test[:, 9:]

    # Attributes encoder (nominal)
    X_train_cat, X_val_cat, X_test_cat = data_encoder(
        train=X_train_cat, val=X_val_cat, test=X_test_cat, path=path, encoder="nominal"
    )

    # Attributes encoder (numerical)
    X_train_num, X_val_num, X_test_num = data_encoder(
        train=X_train_num,
        val=X_val_num,
        test=X_test_num,
        path=path,
        encoder="numerical",
    )

    X_train_cat = np.concatenate((X_train_cat, X_train_bin), axis=1)
    X_train = np.concatenate((X_train_cat, X_train_num), axis=1)

    X_val_cat = np.concatenate((X_val_cat, X_val_bin), axis=1)
    X_val = np.concatenate((X_val_cat, X_val_num), axis=1)

    X_test_cat = np.concatenate((X_test_cat, X_test_bin), axis=1)
    X_test = np.concatenate((X_test_cat, X_test_num), axis=1)

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_sonar_dataset(path, filename, test_size=0.20):

    # Read Dataset
    df = pd.read_csv(path + filename, sep=",")

    df = df.rename(columns={"Class": "class"})

    column_names = list(df.columns)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".txt", "")
    df.to_csv(path + name + ".csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder (numerical)
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_fico_dataset(path, filename, test_size=0.20):

    # Read Dataset
    df = pd.read_csv(path + filename)

    df = df.rename(columns={"RiskPerformance": "class"})

    class_col = df.pop("class")
    df.insert(23, "class", class_col)

    column_names = list(df.columns)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".csv", "")
    df.to_csv(path + name + "_2.csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder (numerical)
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_drybean_dataset(path, filename, test_size=0.20):

    # Read Dataset
    df = pd.read_csv(path + filename)

    df = df.rename(columns={"Class": "class"})

    column_names = list(df.columns)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".csv", "")
    df.to_csv(path + name + "_2.csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder (numerical)
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_avila_dataset(path, filename, test_size=0.20):

    column_names = [str(x) for x in range(1, 11)]
    column_names.append("class")

    test = filename.replace("tr.txt", "ts.txt")

    # Read Dataset
    df_tr = pd.read_csv(path + filename, names=column_names)
    df_ts = pd.read_csv(path + test, names=column_names)

    # Shuffle
    df_tr = df_tr.sample(frac=1).reset_index(drop=True)
    df_ts = df_ts.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace("-tr.txt", "")
    df_tr.to_csv(path + name + "_tr.csv", index=False)
    df_ts.to_csv(path + name + "_ts.csv", index=False)

    n_classes = len(df_tr["class"].unique())
    n_features_in = df_tr.shape[1] - 1

    X = df_tr.values[:, :-1]
    y = df_tr.values[:, -1]

    X_test = df_ts.values[:, :-1]
    y_test = df_ts.values[:, -1]

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder (numerical)
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df_tr": df_tr,
        "df_ts": df_ts,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_banknote_dataset(path, filename, test_size=0.20):

    column_names = [str(x) for x in range(1, 5)]
    column_names.append("class")

    # Read Dataset
    df = pd.read_csv(path + filename, names=column_names)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace("-tr.txt", "")
    df.to_csv(path + "banknote.csv", index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder (numerical)
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_isolet_dataset(path, filename, test_size=0.20):

    column_names = [str(x) for x in range(1, 618)]
    column_names.append("class")

    test = filename.replace("1+2+3+4.data", "5.data")

    # Read Dataset
    df_tr = pd.read_csv(path + filename, names=column_names)
    df_ts = pd.read_csv(path + test, names=column_names)

    # Shuffle
    df_tr = df_tr.sample(frac=1).reset_index(drop=True)
    df_ts = df_ts.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace("1+2+3+4.data", "")
    df_tr.to_csv(path + name + "_tr.csv", index=False)
    df_ts.to_csv(path + name + "_ts.csv", index=False)

    n_classes = len(df_tr["class"].unique())
    n_features_in = df_tr.shape[1] - 1

    X = df_tr.values[:, :-1]
    y = df_tr.values[:, -1]

    X_test = df_ts.values[:, :-1]
    y_test = df_ts.values[:, -1]

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder (numerical)
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df_tr": df_tr,
        "df_ts": df_ts,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_wine_dataset(path, filename, test_size=0.20):

    white = filename.replace("-red.csv", "-white.csv")

    # Read Dataset
    df_red = pd.read_csv(path + filename, sep=";")
    df_white = pd.read_csv(path + white, sep=";")

    df_red = df_red.rename(columns={"quality": "class"})
    df_white = df_white.rename(columns={"quality": "class"})

    df = pd.concat([df_red, df_white], ignore_index=True)

    column_names = list(df.columns)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace("-red.csv", ".csv")
    df.to_csv(path + name, index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder (numerical)
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_yeast_dataset(path, filename, test_size=0.20):

    column_names = [str(x) for x in range(0, 9)]
    column_names.append("class")

    # Read Dataset
    df = pd.read_csv(path + filename, names=column_names, sep="  ", engine="python")

    df = df.drop("0", axis=1)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".data", ".csv")
    df.to_csv(path + name, index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_glass_dataset(path, filename, test_size=0.20):

    column_names = [str(x) for x in range(0, 10)]
    column_names.append("class")

    # Read Dataset
    df = pd.read_csv(path + filename, names=column_names, sep=",")

    df = df.drop("0", axis=1)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".data", ".csv")
    df.to_csv(path + name, index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder (numerical)
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_vehicle_dataset(path, filename, test_size=0.20):

    column_names = [str(x) for x in range(1, 19)]
    column_names.append("class")

    # Read Dataset
    df_xaa = pd.read_csv(
        path + "xaa" + filename, names=column_names, delimiter=" ", index_col=False
    )
    df_xab = pd.read_csv(
        path + "xab" + filename, names=column_names, delimiter=" ", index_col=False
    )
    df_xac = pd.read_csv(
        path + "xac" + filename, names=column_names, delimiter=" ", index_col=False
    )
    df_xad = pd.read_csv(
        path + "xad" + filename, names=column_names, delimiter=" ", index_col=False
    )
    df_xae = pd.read_csv(
        path + "xae" + filename, names=column_names, delimiter=" ", index_col=False
    )
    df_xaf = pd.read_csv(
        path + "xaf" + filename, names=column_names, delimiter=" ", index_col=False
    )
    df_xag = pd.read_csv(
        path + "xag" + filename, names=column_names, delimiter=" ", index_col=False
    )
    df_xah = pd.read_csv(
        path + "xah" + filename, names=column_names, delimiter=" ", index_col=False
    )
    df_xai = pd.read_csv(
        path + "xai" + filename, names=column_names, delimiter=" ", index_col=False
    )

    df_xa = pd.concat([df_xaa, df_xab], ignore_index=True)
    df_xa = pd.concat([df_xa, df_xac], ignore_index=True)
    df_xa = pd.concat([df_xa, df_xad], ignore_index=True)
    df_xa = pd.concat([df_xa, df_xae], ignore_index=True)
    df_xa = pd.concat([df_xa, df_xaf], ignore_index=True)
    df_xa = pd.concat([df_xa, df_xag], ignore_index=True)
    df_xa = pd.concat([df_xa, df_xah], ignore_index=True)
    df = pd.concat([df_xa, df_xai], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".dat", "vehicle.csv")
    df.to_csv(path + name, index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder (numerical)
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_egg_dataset(path, filename, test_size=0.20):

    raw_data = loadarff(path + filename)
    df = pd.DataFrame(raw_data[0])

    df = df.rename(columns={"eyeDetection": "class"})
    column_names = list(df.columns)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save it in csv
    name = filename.replace(".arff", ".csv")
    df.to_csv(path + name, index=False)

    n_classes = len(df["class"].unique())
    n_features_in = df.shape[1] - 1

    # Split into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        df=df, path=path, test_size=test_size
    )

    # Label encoder
    y_train, y_val, y_test = data_encoder(
        train=y_train, val=y_val, test=y_test, path=path, encoder="label"
    )

    # Attributes encoder (numerical)
    X_train, X_val, X_test = data_encoder(
        train=X_train, val=X_val, test=X_test, path=path, encoder="numerical"
    )

    # Save datasets
    save_dataset(
        path=path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    dataset = {
        "name": name,
        "df": df,
        "columns": column_names,
        "n_classes": n_classes,
        "n_features_in": n_features_in,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    return dataset


def prepare_all_datasets():
    paths = {
        "iris": "../datasets/iris/",
        "car": "../datasets/car/",
        "pima": "../datasets/pima/",
        "australian": "../datasets/australian/",
        "lymph": "../datasets/lymph/",
        "breast": "../datasets/breast/",
        "heart": "../datasets/heart/",
        "ecoli": "../datasets/ecoli/",
        "led7": "../datasets/led7/",
        "bank": "../datasets/bank/",
        "adult": "../datasets/adult/",
        "compas": "../datasets/compas/",
        "german": "../datasets/german/",
        "churn": "../datasets/churn/",
        "sonar": "../datasets/sonar/",
        "fico": "../datasets/fico/",
        "drybean": "../datasets/drybean/",
        "avila": "../datasets/avila/",
        "banknote": "../datasets/banknote/",
        "isolet": "../datasets/isolet/",
        "wine": "../datasets/wine/",
        "yeast": "../datasets/yeast/",
        "glass": "../datasets/glass/",
        "vehicle": "../datasets/vehicle/",
        "egg": "../datasets/egg/",
    }

    filenames = {
        "iris": "iris.data",
        "car": "car.data",
        "pima": "pima-diabetes.csv",
        "australian": "australian.dat",
        "lymph": "lymph.txt",
        "breast": "breast.data",
        "heart": "heart-statlog.dat",
        "ecoli": "ecoli.data",
        "led7": "led7.txt",
        "bank": "bank.txt",
        "adult": "adult.txt",
        "compas": "compas.txt",
        "german": "german.data-numeric",
        "churn": "churn_origin.csv",
        "sonar": "sonar.txt",
        "fico": "fico.csv",
        "drybean": "drybean.csv",
        "avila": "avila-tr.txt",
        "banknote": "data_banknote_authentication.txt",
        "isolet": "isolet1+2+3+4.data",
        "wine": "winequality-red.csv",
        "yeast": "yeast.data",
        "glass": "glass.data",
        "vehicle": ".dat",
        "egg": "egg.arff",
    }

    prepare_iris_dataset(path=paths["iris"], filename=filenames["iris"], test_size=0.20)
    prepare_car_dataset(path=paths["car"], filename=filenames["car"], test_size=0.20)
    prepare_pima_dataset(path=paths["pima"], filename=filenames["pima"], test_size=0.20)
    prepare_australian_dataset(
        path=paths["australian"], filename=filenames["australian"], test_size=0.20
    )
    prepare_lymph_dataset(
        path=paths["lymph"], filename=filenames["lymph"], test_size=0.20
    )
    prepare_breast_dataset(
        path=paths["breast"], filename=filenames["breast"], test_size=0.20
    )
    prepare_heart_dataset(
        path=paths["heart"], filename=filenames["heart"], test_size=0.30
    )
    prepare_ecoli_dataset(
        path=paths["ecoli"], filename=filenames["ecoli"], test_size=0.30
    )
    prepare_led7_dataset(path=paths["led7"], filename=filenames["led7"], test_size=0.30)
    prepare_bank_dataset(path=paths["bank"], filename=filenames["bank"], test_size=0.20)
    prepare_adult_dataset(
        path=paths["adult"], filename=filenames["adult"], test_size=0.20
    )
    prepare_compas_dataset(
        path=paths["compas"], filename=filenames["compas"], test_size=0.20
    )
    prepare_german_dataset(
        path=paths["german"], filename=filenames["german"], test_size=0.20
    )
    prepare_churn_dataset(
        path=paths["churn"], filename=filenames["churn"], test_size=0.20
    )
    prepare_sonar_dataset(
        path=paths["sonar"], filename=filenames["sonar"], test_size=0.20
    )
    prepare_fico_dataset(path=paths["fico"], filename=filenames["fico"], test_size=0.20)
    prepare_drybean_dataset(
        path=paths["drybean"], filename=filenames["drybean"], test_size=0.20
    )
    prepare_avila_dataset(
        path=paths["avila"], filename=filenames["avila"], test_size=0.20
    )
    prepare_banknote_dataset(
        path=paths["banknote"], filename=filenames["banknote"], test_size=0.20
    )
    prepare_isolet_dataset(
        path=paths["isolet"], filename=filenames["isolet"], test_size=0.20
    )
    prepare_wine_dataset(path=paths["wine"], filename=filenames["wine"], test_size=0.20)
    prepare_yeast_dataset(
        path=paths["yeast"], filename=filenames["yeast"], test_size=0.20
    )
    prepare_glass_dataset(
        path=paths["glass"], filename=filenames["glass"], test_size=0.20
    )
    prepare_vehicle_dataset(
        path=paths["vehicle"], filename=filenames["vehicle"], test_size=0.20
    )
    prepare_egg_dataset(path=paths["egg"], filename=filenames["egg"], test_size=0.20)


if __name__ == "__main__":
    prepare_all_datasets()
