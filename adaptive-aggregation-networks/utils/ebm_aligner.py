import torch
import torch.nn as nn


class EBMAligner:
    """Manages the lifecycle of the proposed Energy Based Latent Alignment.
    """
    def __init__(self):
        self.is_enabled = False

        # Configs of the EBM model
        self.ebm_latent_dim = 64
        self.ebm_n_layers = 2
        self.ebm_n_hidden_layers = 64
        self.ebm_ema = None

        # EBM Learning configs
        self.max_iter = 5000
        self.ebm_lr = 0.0001
        self.n_langevin_steps = 30
        self.langevin_lr = 0.1
        self.ema_decay = 0.89

        # EBM Loss config
        self.alpha = 0.1

    def ema(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())
        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def sampler(self, ebm_model, x, langevin_steps=30, lr=0.1, return_seq=False):
        """The langevin sampler to sample from the ebm_model

        :param ebm_model: The source EBM model
        :param x: The data which is updated to minimize energy from EBM
        :param langevin_steps: The number of langevin steps
        :param lr: The langevin learning rate
        :param return_seq: Whether to return the sequence of updates to x
        :return: Sample(s) from EBM
        """
        x = x.clone().detach()
        x.requires_grad_(True)
        sgd = torch.optim.SGD([x], lr=lr)

        sequence = torch.zeros_like(x).unsqueeze(0).repeat(langevin_steps, 1, 1)
        for k in range(langevin_steps):
            sequence[k] = x.data
            ebm_model.zero_grad()
            sgd.zero_grad()
            energy = ebm_model(x).sum()

            (-energy).backward()
            sgd.step()

        if return_seq:
            return sequence
        else:
            return x.clone().detach()

    def learn_ebm(self, prev_model, current_model, current_task_data, validation_data=None):
        """Learn the EBM.

        current_task_data + prev_model acts as in-distribution data, and
        current_task_data + current_model acts as out-of-distribution data.
        This is used for learning the energy manifold.

        :param prev_model: Model trained till previous task.
        :param current_model: Model trained on current task.
        :param current_task_data: Datapoints from the current incremental task.
        :param validation_data: OPTIONAL, if passed, used for evaluation.
        :return: None.
        """
        ebm = EBM(latent_dim=self.ebm_latent_dim, n_layer=self.ebm_n_layers,
                       n_hidden=self.ebm_n_hidden_layers).cuda()
        # if self.ebm_ema is None:
        self.ebm_ema = EBM(latent_dim=self.ebm_latent_dim, n_layer=self.ebm_n_layers,
                           n_hidden=self.ebm_n_hidden_layers).cuda()
        # Initialize the exponential moving average of the EBM.
        self.ema(self.ebm_ema, ebm, decay=0.)

        ebm_optimizer = torch.optim.RMSprop(ebm.parameters(), lr=self.ebm_lr)

        iterations = 0
        prev_model.eval()
        current_model.eval()
        data_iter = iter(current_task_data)

        print('Starting to learn the EBM')
        while iterations < self.max_iter:
            ebm.zero_grad()
            ebm_optimizer.zero_grad()

            try:
                inputs, _ = next(data_iter)
            except (OSError, StopIteration):
                data_iter = iter(current_task_data)
                inputs, _ = next(data_iter)

            inputs = inputs.cuda()
            _, prev_z = prev_model(inputs, return_z_also=True)
            _, current_z = current_model(inputs, return_z_also=True)

            self.requires_grad(ebm, False)
            sampled_z = self.sampler(ebm, current_z.clone().detach(), langevin_steps=self.n_langevin_steps, lr=self.langevin_lr)
            self.requires_grad(ebm, True)

            indistribution_energy = ebm(prev_z)
            oodistribution_energy = ebm(sampled_z)

            loss = -(indistribution_energy - oodistribution_energy).mean()

            loss.backward()
            ebm_optimizer.step()
            self.ema(self.ebm_ema, ebm, decay=self.ema_decay)

            if iterations == 0 or iterations % 1000 == 0:
                if validation_data is not None:
                    accuracy = self.evaluate(prev_model, current_model, validation_data)
                    print("Iteration: {:5d}, loss: {:7.2f}, accuracy: {:5.2f}".format(iterations, loss, accuracy))
                else:
                    print("Iter: {:5d}, loss: {:7.2f}".format(iterations, loss))

            iterations += 1

        self.is_enabled = True

    def evaluate(self, previous_model, current_model, validation_data):
        previous_model.eval()
        current_model.eval()
        accuracy_metric = Metrics()

        for inputs, labels in validation_data:
            inputs = inputs.cuda()
            labels = labels.cuda()
            _, current_z = current_model(inputs, return_z_also=True)
            aligned_z = self.align_latents(current_z)

            output = previous_model.fc(aligned_z)
            accuracy = self.compute_accuracy(output, labels)[0].item()
            accuracy_metric.update(accuracy)

        return accuracy_metric.avg

    def compute_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result

    def align_latents(self, z):
        self.requires_grad(self.ebm_ema, False)
        aligned_z = self.sampler(self.ebm_ema, z.clone().detach(), langevin_steps=self.n_langevin_steps, lr=self.langevin_lr)
        self.requires_grad(self.ebm_ema, True)
        return aligned_z

    def loss(self, z):
        aligned_z = self.align_latents(z).clone().detach()
        loss = self.alpha * nn.MSELoss()(z, aligned_z)
        return loss


class EBM(nn.Module):
    """Defining the Energy Based Model.
    """
    def __init__(self, latent_dim=32, n_layer=1, n_hidden=64):
        super().__init__()

        mlp = nn.ModuleList()
        if n_layer == 0:
            mlp.append(nn.Linear(latent_dim, 1))
        else:
            mlp.append(nn.Linear(latent_dim, n_hidden))

            for _ in range(n_layer-1):
                mlp.append(nn.LeakyReLU(0.2))
                mlp.append(nn.Linear(n_hidden, n_hidden))

            mlp.append(nn.LeakyReLU(0.2))
            mlp.append(nn.Linear(n_hidden, 1))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


class Metrics:
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count