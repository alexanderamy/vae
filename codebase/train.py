import numpy as np
import torch
from torch import optim
from codebase import utils as ut

def train(model, train_loader, device, tqdm,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', y_status='none', reinitialize=False):
    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    i = 0
    with tqdm(total=iter_max) as pbar:
        while True:
            for batch_idx, (xu, yu) in enumerate(train_loader):
                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()

                xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                yu = yu.new(np.eye(10)[yu]).to(device).float()
                loss = model.loss(xu)
                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    loss='{:.2e}'.format(loss))
                pbar.update(1)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i == iter_max:
                    return
