import os
import logging
import datetime


def get_local_time():
    return datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")


class Logger(object):
    def __init__(self, model_name="LightGCN", dataset_name="dataset"):
        log_dir_path = "./output/log/{}".format(model_name)

        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        self.logger = logging.getLogger("train_logger")
        self.logger.setLevel(logging.INFO)
        # A fresh handler per Logger, otherwise repeated construction in one
        # process would duplicate every line.
        self.logger.handlers.clear()
        log_file = logging.FileHandler(
            "{}/{}_{}.log".format(log_dir_path, dataset_name, get_local_time()),
            "a",
            encoding="utf-8",
        )
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)

    def log(self, message, save_to_log=True, print_to_console=True):
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_loss(
        self, epoch_idx, loss_log_dict, save_to_log=True, print_to_console=True
    ):
        message = "[Epoch {:3d}] ".format(epoch_idx)
        for loss_name in loss_log_dict:
            message += "{}: {:.4f} ".format(loss_name, loss_log_dict[loss_name])
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_eval(
        self,
        eval_result,
        k,
        data_type,
        save_to_log=True,
        print_to_console=True,
        epoch_idx=None,
    ):
        if epoch_idx is not None:
            message = "Epoch {:3d} {} ".format(epoch_idx, data_type)
        else:
            message = ""

        for metric in eval_result:
            message += "["
            for i in range(len(k)):
                message += "{}@{}: {:.4f} ".format(metric, k[i], eval_result[metric][i])
            message += "] "
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)
