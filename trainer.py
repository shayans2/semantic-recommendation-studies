import os
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.optim as optim
from metrics import Metric


class Trainer(object):
    def __init__(
        self,
        data_handler,
        epoch,
        lr,
        weight_decay,
        batch_size,
        patience,
        logger,
        k,
        device,
        model_name,
        dataset_name,
        model,
        save_checkpoint=True,
        test_at_end=True,
    ):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric(k=k, device=device)
        self.epoch = epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.k = k
        self.device = device
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model = model
        self.save_checkpoint_at_end = save_checkpoint
        self.test_at_end = test_at_end
        self.best_epoch = None
        self.best_metric_value = None
        self.stopped_reason = "epoch_cap"

    def create_optimizer(self, model, optimizer_type="adam"):
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

    # def create_optimizer(self, model):
    #     self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_epoch(self, model, epoch_idx):
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # SIG resamples its denoised graph each epoch. Inert for models
        # that do not define the hook, so baseline behaviour is unchanged.
        if hasattr(model, "on_epoch_start"):
            model.on_epoch_start()

        loss_log_dict = {}
        ep_loss = 0
        model.train()

        for _, tem in tqdm(
            enumerate(train_dataloader),
            desc="Training Recommender",
            total=len(train_dataloader),
        ):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(self.device), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # Record per-batch loss
            for loss_name, loss_val in loss_dict.items():
                if torch.is_tensor(loss_val):
                    loss_val = loss_val.detach()
                loss_log_dict[loss_name] = loss_log_dict.get(loss_name, 0) + float(
                    loss_val
                )

        # Parameters changed this epoch, so any cached propagation is stale.
        model.reset_cache()

        # Normalize accumulated loss
        for loss_name in loss_log_dict:
            loss_log_dict[loss_name] /= len(train_dataloader)

        self.logger.log_loss(epoch_idx, loss_log_dict)

    def train(self, model, monitor_metric="recall", eval_every=3):
        now_patience = 0
        best_epoch = 0
        best_metric_value = -1e9
        best_state_dict = deepcopy(model.state_dict())
        self.create_optimizer(model)

        for epoch_idx in range(self.epoch):
            self.train_epoch(model, epoch_idx)

            if epoch_idx % eval_every == 0:
                eval_result = self.evaluate(model, epoch_idx)

                metric_value = eval_result[monitor_metric][-1]
                if metric_value > best_metric_value:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_metric_value = metric_value
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    now_patience += 1

                if now_patience == self.patience:
                    self.stopped_reason = "early_stopping"
                    break

        # Restore the best checkpoint. reset_cache() is essential: final_embeds
        # is a plain attribute, not part of state_dict, so without it evaluation
        # would silently keep serving the last epoch's cached embeddings.
        model = self.model
        model.load_state_dict(best_state_dict)
        model.reset_cache()
        self.best_epoch = best_epoch
        self.best_metric_value = best_metric_value

        # evaluation again
        self.evaluate(model)

        # A fixed final comparison may test here. Hyperparameter searches can
        # disable this so the held-out test set remains unseen during tuning.
        test_result = self.test(model) if self.test_at_end else None

        if self.save_checkpoint_at_end:
            self.save_model(model)
        self.logger.log("Best Epoch {}. Final test result: {}.".format(
            best_epoch, test_result if self.test_at_end else "not evaluated"
        ))

    def evaluate(self, model, epoch_idx=None):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
        self.logger.log_eval(
            eval_result, self.k, data_type="Validation set", epoch_idx=epoch_idx
        )
        return eval_result

    def test(self, model):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, self.k, data_type="Test set")
        return eval_result

    def test_save(self, model):
        model.eval()
        eval_result, candidate_set = self.metric.eval_save(
            model, self.data_handler.test_dataloader
        )
        self.logger.log_eval(eval_result, self.k, data_type="Test set")
        return eval_result, candidate_set

    def save_model(self, model):
        model_state_dict = model.state_dict()
        model_name = self.model_name
        save_dir_path = "./output/checkpoint/{}".format(model_name)
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        torch.save(
            model_state_dict,
            "{}/{}-{}-{}.pth".format(
                save_dir_path, model_name, self.dataset_name, "42"
            ),
        )
        self.logger.log(
            "Save model parameters to {}".format(
                "{}/{}-{}-{}.pth".format(
                    save_dir_path, model_name, self.dataset_name, "42"
                )
            )
        )
