import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import env.Config as config

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoints.pth'))
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def save_model(model, model_output_dir, epoch):
    save_model_file = os.path.join(model_output_dir, "epoch_{}.pth".format(epoch))
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    torch.save(model, save_model_file)

class TrainerBase(nn.Module):
    def __init__(self,
                 epoches,
                 train_loader,
                 valid_loader,
                 optimizer,
                 device,
                 IFEarlyStopping,
                 IFadjust_learning_rate,
                 **kwargs):
        super(TrainerBase, self).__init__()

        self.epoches = epoches
        if self.epoches is None:
            raise ValueError("Please provide the total number of training epochs")

        self.valid_loader = valid_loader
        if self.valid_loader is None:
            raise ValueError("Please provide valid_loader")

        self.train_loader = train_loader
        if self.train_loader is None:
            raise ValueError("Please provide train_loader")

        self.optimizer = optimizer
        if self.optimizer is None:
            raise ValueError("Please provide an optimizer")

        self.device = device
        if self.device is None:
            raise ValueError("Please provide the device type")

        # Perform the following checks if early stopping is enabled
        self.IFEarlyStopping = IFEarlyStopping
        if IFEarlyStopping:
            if "patience" in kwargs.keys():
                self.early_stopping = EarlyStopping(patience=kwargs["patience"], verbose=True)
            else:
                raise ValueError("Enabling early stopping requires the {patience=int X} parameter")

            self.val_loader = self.valid_loader

        # Perform the following checks if learning-rate adjustment is enabled
        self.IFadjust_learning_rate = IFadjust_learning_rate
        self.types = None
        self.lr_adjust = None
        if IFadjust_learning_rate:
            if "types" in kwargs.keys():
                self.types = kwargs["types"]
                if "lr_adjust" in kwargs.keys():
                    self.lr_adjust = kwargs["lr_adjust"]
                else:
                    self.lr_adjust = None
            else:
                raise ValueError("Enabling learning-rate adjustment requires selecting type1 or type2 for the types parameter")

    def adjust_learning_rate(self, epoch, learning_rate):
        # lr = args.learning_rate * (0.2 ** (epoch // 2))
        if not self.IFadjust_learning_rate:
            return

        if self.types == 'type1':
            lr_adjust = {epoch: learning_rate * (0.1 ** ((epoch - 1) // 10))}  # Reduce the learning rate tenfold every 10 epochs
        elif self.types == 'type2':
            if self.lr_adjust is not None:
                lr_adjust = self.lr_adjust
            else:
                lr_adjust = {
                    5: 1e-4, 10: 5e-5, 20: 1e-5, 25: 5e-6,
                    30: 1e-6, 35: 5e-7, 40: 1e-8
                }
        else:
            raise ValueError("Please select {0} or {1} for the learning-rate adjustment parameter types".format("type1", "type2"))

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    @staticmethod
    def save_model_btempCond(model, path, i):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, 'btempCond_epoch_' + str(i + 1) + '.pth')
        torch.save(model, save_path)
        print("saving:" + save_path)

    @staticmethod
    def save_model(model, path, i):
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, 'epoch_' + str(i + 1) + '.pth')
        torch.save(model, save_path)
        print("saving:" + save_path)
    def forward(self, model, model_save_path, save_fig_dir):

        pass

def Paint_fig(train_losses, valid_losses, save_fig_dir):
    os.makedirs(save_fig_dir, exist_ok=True)
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
    train_loss, = plt.plot(np.arange(0, len(train_losses)), train_losses, 'r')
    # val_loss, = plt.plot(np.arange(0, len(valid_losses)), valid_losses, 'g')
    plt.xlabel('epochs')
    plt.ylabel('Diffusion loss')
    plt.title("train/valid loss vs epoch")
    # Add the legend
    # ax_loss.legend(handles=[train_loss, val_loss], labels=['train_epoch_loss', 'val_epoch_loss'],
    #                loc='best')
    fig_loss.savefig(os.path.join(save_fig_dir, 'loss.png'))
    plt.close(fig_loss)

def _grad_calculate(task_losses, model_params, eps: float = 1e-8):
    """
    Calculate the asymmetric affinity matrix A (task_count x task_count) across tasks.

    Args:
        task_losses: list of torch.Tensor, where each element is the scalar loss of a subtask,
                     ordered as [loss_lon, loss_lat, loss_press, loss_wind]
        model_params: iterable of torch.nn.Parameter, typically generator.parameters()
        eps:         float, a small constant added to the denominator for numerical stability

    Returns:
        affinity: torch.Tensor of shape (T, T), where T = len(task_losses);
                  affinity[i, j] = (g_i·g_j) / (||g_j||^2 + eps)
    """
    T = len(task_losses)
    # 1. Calculate the gradient vector for each task separately
    #    retain_graph=True preserves the computation graph for repeated grad calls; the final loss
    #    need not preserve it, but True is used consistently here
    grads = []
    for loss in task_losses:
        # allow_unused=True prevents errors when some parameters have no gradients
        g = torch.autograd.grad(
            loss, model_params,
            retain_graph=True, allow_unused=True
        )
        # Concatenate gradient tensors into one long vector, replacing None gradients with zeros
        flat = []
        for grad, p in zip(g, model_params):
            if grad is None:
                flat.append(torch.zeros_like(p).view(-1))
            else:
                flat.append(grad.contiguous().view(-1))
        grads.append(torch.cat(flat, dim=0))  # shape [total_param_dim]
    # 2. Calculate the asymmetric affinity matrix
    affinity = torch.zeros(T, T, device=grads[0].device, dtype=grads[0].dtype)
    for i in range(T):
        for j in range(T):
            dot_ij = torch.dot(grads[i], grads[j])
            norm_j2 = grads[j].pow(2).sum()
            affinity[i, j] = dot_ij / (norm_j2 + eps)
    return affinity
def GroupGradOptLoss(task_losses, shared_params, loss_weight=None):
    aff_mat = _grad_calculate(task_losses, shared_params)

    # aff_mat[i,j] is the affinity from task i to task j
    print("task affinity matrix: \n", aff_mat)
    # logger.info(
    #     "Task affinity matrix:\n", aff_mat
    # )
    # ===== Adaptively adjust multitask weights using the affinity matrix =====
    # aff_mat: [n,n], where each row sum represents how well task i collaborates with other tasks
    #             i.e. w_i_raw = sum_j aff_mat[i,j]
    w_raw = aff_mat.sum(dim=1)  # shape [n]

    # Multiply by the learnable vector on the generator, then activate and normalize
    #w_learn = F.softplus(w_raw * loss_weight)  # [n]
    w_learn = F.softplus(w_raw)  # [n]

    w = w_learn / (w_learn.sum() + 1e-8)  # [n]
    weighted_loss = (w * torch.stack(task_losses)).sum()

    return weighted_loss

class CondDiffusionTrainer_GroupGrad(TrainerBase):
    def __init__(self,
                 epoches=None,
                 train_loader=None,
                 valid_loader=None,
                 optimizer=None,
                 device=None,
                 IFEarlyStopping=False,
                 IFadjust_learning_rate=False,
                 **kwargs):
        super(CondDiffusionTrainer_GroupGrad, self).__init__(epoches, train_loader, valid_loader, optimizer, device,
                                                     IFEarlyStopping, IFadjust_learning_rate,
                                                     **kwargs)

        if "timesteps" in kwargs.keys():
            self.timesteps = kwargs["timesteps"]
        else:
            raise ValueError("diffusion must provide parameter: step")
    def forward(self, model, model_save_path, save_fig_dir):
        train_losses = []
        valid_losses = []
        for i in range(self.epoches):
            train_batch_loss = 0
            train_loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            train_batch_num = 0
            for batch, data in train_loop:
                pr = data["pr"].to(config.device)
                k8_btemp = data["k8_btemp"]
                k8_btemp = k8_btemp.to(config.device)
                pxh_k8_btemp = data["pxh_k8_btemp"]
                pxh_k8_btemp = pxh_k8_btemp.to(config.device)
                btemp = torch.cat([pxh_k8_btemp, k8_btemp], dim=1)

                group1_label = data["group1_label"].to(config.device)  # b, 1, 3
                group2_label = data["group2_label"].to(config.device)  # b, 3, 4
                group2_label_mask = data["group2_label_mask"].to(config.device)  # b, 5, 3

                '''Loss for gradient optimization'''
                # loss_list = model(group1_label, group2_label, group2_label_mask, cond_visual=btemp)
                loss_list = model(group1_label, group2_label, group2_label_mask, cond_visual=btemp, cond_ratio=pr)
                shared_params = []
                # Core network
                # shared_params += list(model.denoise_fn.visual_cond_enc.parameters())
                shared_params += list(model.denoise_fn.cond_enc.parameters())
                shared_params += list(model.denoise_fn.time_mlp.parameters())
                # total_loss = GroupGradOptLoss(loss_list, shared_params, model.loss_weight)
                total_loss = GroupGradOptLoss(loss_list, shared_params)

                train_batch_loss += total_loss.item()
                train_batch_num += 1
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Update the tqdm progress bar
                train_loop.set_description(f'Train Epoch [{i + 1}/{self.epoches}]')
                train_loop.set_postfix(diffusion_loss=total_loss.item())
                # train_loop.set_postfix(noise_loss=noise_loss.item(), prob_loss=prob_loss.item(), diffusion_loss=total_loss.item())
            train_losses.append(train_batch_loss / train_batch_num)
            print("Train Epoch = {} mean loss = {} ".format(i, train_batch_loss / train_batch_num))
            if (i+1) % config.save_model_iter == 0:
                self.save_model(model, model_save_path, i)

        Paint_fig(train_losses, valid_losses, save_fig_dir)
        return model
