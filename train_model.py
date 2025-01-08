#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from model import VQVAE

# def train_model_vqvae(dataloader, epochs=10, lr=0.001):
#     model = MyModel()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0
#         for batch_data, batch_labels in dataloader:
#             optimizer.zero_grad()
#             outputs = model(batch_data)
#             loss = criterion(outputs, batch_labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader)}")

#     return model

# Train model.
def train_model_vqvae(dataloader, epochs=10, lr=0.001):

    device = torch.device("cuda:0")
    use_ema = True
    model_args = {
        "in_channels": 3,
        "num_hiddens": 128,
        "num_downsampling_layers": 2,
        "num_residual_layers": 2,
        "num_residual_hiddens": 32,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "use_ema": use_ema,
        "decay": 0.99,
        "epsilon": 1e-5,
    }
    model = VQVAE(**model_args).to(device)

    beta = 0.25

    # Initialize optimizer.
    train_params = [params for params in model.parameters()]
    lr = 3e-4
    optimizer = optim.Adam(train_params, lr=lr)
    criterion = nn.MSELoss()


    epochs = 7
    eval_every = 100
    best_train_loss = float("inf")
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0
        total_recon_error = 0
        n_train = 0
        for (batch_idx, train_tensors) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = train_tensors[0].to(device)
            out = model(imgs)
            recon_error = criterion(out["x_recon"], imgs) / train_data_variance
            total_recon_error += recon_error.item()
            loss = recon_error + beta * out["commitment_loss"]
            if not use_ema:
                loss += out["dictionary_loss"]

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_train += 1

            if ((batch_idx + 1) % eval_every) == 0:
                print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
                total_train_loss /= n_train
                if total_train_loss < best_train_loss:
                    best_train_loss = total_train_loss

                print(f"total_train_loss: {total_train_loss}")
                print(f"best_train_loss: {best_train_loss}")
                print(f"recon_error: {total_recon_error / n_train}\n")

                total_train_loss = 0
                total_recon_error = 0
                n_train = 0
    return model

