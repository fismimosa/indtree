import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from indtree.tree_dataset_generation import direct_encodings_to_random_forest


class MetaModel(pl.LightningModule):
    def __init__(
        self,
        metamodel,
        model,
        optimizer=optim.Adam,
        loss_feat=nn.CrossEntropyLoss(),
        loss_class=nn.CrossEntropyLoss(),
        loss_thresh=nn.MSELoss(),
        weight_feat=1.0,
        weight_class=1.0,
        weight_thresh=1.0,
        adaptation_steps=3,
        metamodel_params=None,
        optimizer_params=None,
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        rf_val=None,
        rf_log_freq=1,
        tree_log_params=None,
        tree_encoding_type="direct",
        split_train_val=0.5,
    ):
        super().__init__()
        self.model = model
        self.metamodel_params = metamodel_params
        if self.metamodel_params is None:
            self.metamodel_params = dict(lr=1e-5)

        self.optimizer_params = optimizer_params
        if optimizer_params is None:
            self.optimizer_params = dict(lr=1e-4)

        self.loss_fn_feat = loss_feat
        self.loss_fn_class = loss_class
        self.loss_fn_thresh = loss_thresh
        self.adaptation_steps = adaptation_steps
        self.weight_feat = weight_feat
        self.weight_class = weight_class
        self.weight_thresh = weight_thresh

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.rf_val = rf_val
        self.rf_log_freq = rf_log_freq
        self.tree_log_params = tree_log_params
        if self.tree_log_params is None:
            self.tree_log_params = dict(
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        self.tree_encoding_type = tree_encoding_type
        self.split_train_val = split_train_val

        self.automatic_optimization = False  # Disable automatic optimization

        self.optimizer_ = optimizer

        self.metamodel_ = metamodel(self.model, **self.metamodel_params)

        self.test_outputs_ = []
        self.all_feat_out_ = None
        self.all_class_out_ = None
        self.all_thresh_out_ = None

        # self.save_hyperparameters()

    def forward(self, model_input, feat_out, thresh_out, class_out):
        (
            adaptation_val_loss,
            adaptation_val_loss_feat,
            adaptation_val_loss_thresh,
            adaptation_val_loss_class,
            feat_out,
            class_out,
            thresh_out,
        ) = self._inner_loop_batch_inference(
            model_input=model_input,
            onehot_feat=feat_out,
            onehot_thresh=thresh_out,
            onehot_class=class_out,
        )
        return feat_out, thresh_out, class_out

    def _compute_metaloss(
        self, onehot_feat, onehot_thresh, onehot_class, feat_out, thresh_out, class_out
    ):

        leaf_idxs = torch.where((onehot_feat.sum(dim=2) == 0).ravel())[0]
        internal_idxs = torch.where((onehot_feat.sum(dim=2) != 0).ravel())[0]

        if leaf_idxs.shape[0] == 0:  # No leaf nodes
            loss_class = torch.tensor(0.0) * class_out.sum()
        else:
            loss_class = self.loss_fn_class(
                class_out.reshape(-1, class_out.shape[2])[leaf_idxs],
                torch.argmax(onehot_class.reshape(-1, onehot_class.shape[2]), dim=1)[
                    leaf_idxs
                ],
            )
        if internal_idxs.shape[0] == 0:
            loss_feat = torch.tensor(0.0) * feat_out.sum()
            loss_thresh = torch.tensor(0.0) * thresh_out.sum()
        else:
            loss_feat = self.loss_fn_feat(
                feat_out.reshape(-1, feat_out.shape[2])[internal_idxs],
                torch.argmax(onehot_feat.reshape(-1, onehot_feat.shape[2]), dim=1)[
                    internal_idxs
                ],
            )
            loss_thresh = self.loss_fn_thresh(
                thresh_out[0][internal_idxs], onehot_thresh[0][internal_idxs]
            )
        total_loss = (
            loss_feat * self.weight_feat
            + loss_class * self.weight_class
            + loss_thresh * self.weight_thresh
        )
        return total_loss, loss_feat, loss_class, loss_thresh

    def _inner_loop(self, model_input, onehot_feat, onehot_thresh, onehot_class):
        learner = self.metamodel_.clone()
        learner.train()
        for _ in range(self.adaptation_steps):
            feat_out, class_out, thresh_out, _ = learner(model_input)
            (
                final_adaptation_error,
                loss_feat,
                loss_class,
                loss_thresh,
            ) = self._compute_metaloss(
                onehot_feat=onehot_feat,
                onehot_thresh=onehot_thresh,
                onehot_class=onehot_class,
                feat_out=feat_out,
                thresh_out=thresh_out,
                class_out=class_out,
            )
            learner.adapt(final_adaptation_error)
        return learner, final_adaptation_error, loss_feat, loss_class, loss_thresh

    def _inner_loop_train(self, model_input, onehot_feat, onehot_thresh, onehot_class):
        n_coords = model_input.shape[1]
        split = int(n_coords * self.split_train_val)
        permutation = torch.randperm(n_coords)
        idxs_train = permutation[:split]
        idxs_val = permutation[split:]
        adapted_learner, _, _, _, _ = self._inner_loop(
            model_input=model_input[:, idxs_train],
            onehot_feat=onehot_feat[:, idxs_train],
            onehot_thresh=onehot_thresh[:, idxs_train],
            onehot_class=onehot_class[:, idxs_train],
        )
        feat_out, class_out, thresh_out, _ = adapted_learner(model_input)
        (
            adaptation_val_loss,
            adaptation_val_loss_feat,
            adaptation_val_loss_thresh,
            adaptation_val_loss_class,
        ) = self._compute_metaloss(
            onehot_feat=onehot_feat[:, idxs_val],
            onehot_thresh=onehot_thresh[:, idxs_val],
            onehot_class=onehot_class[:, idxs_val],
            feat_out=feat_out[:, idxs_val],
            thresh_out=thresh_out[:, idxs_val],
            class_out=class_out[:, idxs_val],
        )
        return (
            adapted_learner,
            adaptation_val_loss,
            adaptation_val_loss_feat,
            adaptation_val_loss_thresh,
            adaptation_val_loss_class,
            feat_out,
            class_out,
            thresh_out,
        )

    def _inner_loop_inference(
        self, model_input, onehot_feat, onehot_thresh, onehot_class
    ):
        adapted_learner, _, _, _, _ = self._inner_loop(
            model_input=model_input,
            onehot_feat=onehot_feat,
            onehot_thresh=onehot_thresh,
            onehot_class=onehot_class,
        )
        feat_out, class_out, thresh_out, _ = adapted_learner(model_input)
        (
            adaptation_val_loss,
            adaptation_val_loss_feat,
            adaptation_val_loss_thresh,
            adaptation_val_loss_class,
        ) = self._compute_metaloss(
            onehot_feat=onehot_feat,
            onehot_thresh=onehot_thresh,
            onehot_class=onehot_class,
            feat_out=feat_out,
            thresh_out=thresh_out,
            class_out=class_out,
        )
        return (
            adapted_learner,
            adaptation_val_loss,
            adaptation_val_loss_feat,
            adaptation_val_loss_thresh,
            adaptation_val_loss_class,
            feat_out,
            class_out,
            thresh_out,
        )

    def _inner_loop_batch_inference(
        self, model_input, onehot_feat, onehot_thresh, onehot_class
    ):
        feat_out = list()
        class_out = list()
        thresh_out = list()

        total_loss = 0
        total_loss_feat = 0
        total_loss_class = 0
        total_loss_thresh = 0

        for i in range(model_input.shape[0]):
            (
                _,
                final_adaptation_loss,
                loss_feat,
                loss_class,
                loss_thresh,
                feat_out_,
                class_out_,
                thresh_out_,
            ) = self._inner_loop_inference(
                model_input=model_input[i : i + 1],
                onehot_feat=onehot_feat[i : i + 1],
                onehot_thresh=onehot_thresh[i : i + 1],
                onehot_class=onehot_class[i : i + 1],
            )
            total_loss += final_adaptation_loss.item() / model_input.shape[0]
            total_loss_feat += loss_feat.item() / model_input.shape[0]
            total_loss_class += loss_class.item() / model_input.shape[0]
            total_loss_thresh += loss_thresh.item() / model_input.shape[0]
            feat_out.append(feat_out_.detach())
            class_out.append(class_out_.detach())
            thresh_out.append(thresh_out_.detach())
        return (
            total_loss,
            total_loss_feat,
            total_loss_class,
            total_loss_thresh,
            torch.cat(feat_out, dim=0),
            torch.cat(class_out, dim=0),
            torch.cat(thresh_out, dim=0),
        )

    def _inner_loop_batch_train(
        self, model_input, onehot_feat, onehot_thresh, onehot_class
    ):
        opt = self.optimizers()
        opt.zero_grad()

        feat_out = list()
        class_out = list()
        thresh_out = list()

        total_loss = 0
        total_loss_feat = 0
        total_loss_class = 0
        total_loss_thresh = 0
        for i in range(model_input.shape[0]):
            (
                _,
                final_adaptation_loss,
                loss_feat,
                loss_class,
                loss_thresh,
                feat_out_,
                class_out_,
                thresh_out_,
            ) = self._inner_loop_train(
                model_input=model_input[i : i + 1],
                onehot_feat=onehot_feat[i : i + 1],
                onehot_thresh=onehot_thresh[i : i + 1],
                onehot_class=onehot_class[i : i + 1],
            )
            self.manual_backward(final_adaptation_loss / model_input.shape[0])
            total_loss += final_adaptation_loss.item() / model_input.shape[0]
            total_loss_feat += loss_feat.item() / model_input.shape[0]
            total_loss_class += loss_class.item() / model_input.shape[0]
            total_loss_thresh += loss_thresh.item() / model_input.shape[0]
            feat_out.append(feat_out_.detach())
            class_out.append(class_out_.detach())
            thresh_out.append(thresh_out_.detach())
        opt.step()
        return (
            total_loss,
            total_loss_feat,
            total_loss_class,
            total_loss_thresh,
            torch.cat(feat_out, dim=0),
            torch.cat(class_out, dim=0),
            torch.cat(thresh_out, dim=0),
        )

    def training_step(self, batch, batch_idx):
        model_input, onehot_feat, onehot_thresh, onehot_class = batch
        (
            total_loss,
            total_loss_feat,
            total_loss_class,
            total_loss_thresh,
            feat_out,
            class_out,
            thresh_out,
        ) = self._inner_loop_batch_train(
            model_input=model_input,
            onehot_feat=onehot_feat,
            onehot_thresh=onehot_thresh,
            onehot_class=onehot_class,
        )

        if self.current_epoch % self.rf_log_freq == 0:
            self._log_tree_metrics(feat_out, thresh_out, class_out, prefix="train")

        self._log_recon_std(feat_out, thresh_out, class_out, prefix="train")

        self._log_loss(
            total_loss,
            total_loss_feat,
            total_loss_thresh,
            total_loss_class,
            prefix="train",
        )

        return  # total_loss

    def _log_loss(self, total_loss, loss_feat, loss_thresh, loss_class, prefix=""):
        self.log(
            prefix + "_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            prefix + "_loss_feat",
            loss_feat,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            prefix + "_loss_class",
            loss_class,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            prefix + "_loss_thresh",
            loss_thresh,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def _log_recon_std(self, feat_out, thresh_out, class_out, prefix=""):
        feat_std = torch.std(feat_out.reshape(feat_out.shape[0], -1), dim=0).mean()
        thresh_std = torch.std(
            thresh_out.reshape(thresh_out.shape[0], -1), dim=0
        ).mean()
        class_std = torch.std(class_out.reshape(class_out.shape[0], -1), dim=0).mean()
        self.log(
            prefix + "_feat_std",
            feat_std,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            prefix + "_thresh_std",
            thresh_std,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            prefix + "_class_std",
            class_std,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def _log_tree_metrics(self, feat_out, thresh_out, class_out, prefix=""):
        # Compute tree metrics
        if self.current_epoch % self.rf_log_freq == 0:
            rf_recon = direct_encodings_to_random_forest(
                feats_embeddings=feat_out.detach().cpu(),
                classes_embeddings=class_out.detach().cpu(),
                thresholds_embeddings=thresh_out.detach().cpu(),
                original_rf=self.rf_val,
            )
            dt_val_scores_train = list()
            dt_val_scores_val = list()
            dt_val_scores_test = list()
            for estimator in rf_recon.estimators_:
                dt_val_scores_train.append(estimator.score(self.X_train, self.y_train))
                dt_val_scores_val.append(estimator.score(self.X_val, self.y_val))
                dt_val_scores_test.append(estimator.score(self.X_test, self.y_test))

            rf_val_score_train = rf_recon.score(self.X_train, self.y_train)
            rf_val_score_val = rf_recon.score(self.X_val, self.y_val)
            rf_val_score_test = rf_recon.score(self.X_test, self.y_test)
            self.log(
                prefix + "_dt_scores_mean_train",
                np.mean(dt_val_scores_train),
                **self.tree_log_params,
            )
            self.log(
                prefix + "_dt_scores_mean_val",
                np.mean(dt_val_scores_val),
                **self.tree_log_params,
            )
            self.log(
                prefix + "_dt_scores_mean_test",
                np.mean(dt_val_scores_test),
                **self.tree_log_params,
            )
            self.log(
                prefix + "_dt_scores_std_train",
                np.std(dt_val_scores_train),
                **self.tree_log_params,
            )
            self.log(
                prefix + "_dt_scores_std_val",
                np.std(dt_val_scores_val),
                **self.tree_log_params,
            )
            self.log(
                prefix + "_dt_scores_std_test",
                np.std(dt_val_scores_test),
                **self.tree_log_params,
            )
            self.log(
                prefix + "_rf_score_train",
                rf_val_score_train,
                **self.tree_log_params,
            )
            self.log(
                prefix + "_rf_score_val",
                rf_val_score_val,
                **self.tree_log_params,
            )
            self.log(
                prefix + "_rf_score_test",
                rf_val_score_test,
                **self.tree_log_params,
            )

    def configure_optimizers(self):
        optimizer = self.optimizer_(
            self.metamodel_.parameters(), **self.optimizer_params
        )
        return optimizer


class TreeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        meta_dataloader_train,
        meta_dataloader_val,
        meta_dataloader_test,
        batch_size,
    ):
        super().__init__()
        self.meta_dataloader_train = meta_dataloader_train
        self.meta_dataloader_val = meta_dataloader_val
        self.meta_dataloader_test = meta_dataloader_test
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = ConcatDataset(self.meta_dataloader_train)
        self.val_dataset = ConcatDataset(self.meta_dataloader_val)
        self.test_dataset = ConcatDataset(self.meta_dataloader_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
