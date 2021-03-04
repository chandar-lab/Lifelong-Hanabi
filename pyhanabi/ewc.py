# Implements EWC loss

import torch
import torch.nn as nn


class EWC(nn.Module):
    """
    Estimates the fisher matrix and calculates ewc loss based on that.
    Args:
        args.ewc_gamma (online EWC): decay-term for old tasks' contribution to quadratic term
        args.online: Bool "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        args.batchsize: int
        args.pred_weight: 0.0 (Auxilary task loss coefficient)
        args.train_device: cuda:0
    Returns:
        float: ewc loss
    """

    def __init__(self, args):
        super().__init__()
        self.ewc_gamma = args.ewc_gamma
        self.online = args.online
        self.batchsize = args.batchsize
        self.pred_weight = args.pred_weight
        self.train_device = args.train_device

    def forward(self, x):
        pass

        # ----------------- EWC-specifc functions -----------------#

    def estimate_fisher(self, learnable_agent, batch, weight, stat, task_idx):
        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}

        tmp_loss, tmp_priority = learnable_agent.loss(batch, self.pred_weight, stat)
        tmp_loss = (tmp_loss * weight).mean()
        tmp_loss.backward()

        # Square gradients and keep running sum
        for n, p in learnable_agent.online_net.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                est_fisher_info[n] = p.detach().clone().zero_()
                if p.grad is not None:
                    est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p / self.batchsize for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in learnable_agent.online_net.named_parameters():
            if p.requires_grad:
                n = n.replace(".", "__")
                self.register_buffer(
                    "{}_EWC_prev_task{}".format(n, "" if self.online else task_idx + 1),
                    p.detach().clone(),
                )
                if self.online and task_idx > 0:
                    existing_values = getattr(self, "{}_EWC_estimated_fisher".format(n))
                    est_fisher_info[n] += self.ewc_gamma * existing_values
                self.register_buffer(
                    "{}_EWC_estimated_fisher{}".format(
                        n, "" if self.online else task_idx + 1
                    ),
                    est_fisher_info[n],
                )
        return tmp_priority

    def compute_ewc_loss(self, learnable_agent, task_idx):
        if task_idx > 0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, task_idx + 1):
                for n, p in learnable_agent.online_net.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace(".", "__")
                        mean = getattr(
                            self,
                            "{}_EWC_prev_task{}".format(n, "" if self.online else task),
                        )
                        fisher = getattr(
                            self,
                            "{}_EWC_estimated_fisher{}".format(
                                n, "" if self.online else task
                            ),
                        )
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.ewc_gamma * fisher if self.online else fisher

                        # Calculate EWC-loss
                        losses.append((fisher * (p - mean) ** 2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1.0 / 2) * sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0.0, device=self.train_device)
