import csv

import torch
import torch._dynamo
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets import EvalDataset, TrainDataset
from Gloss import SSIM, GradientVariance
from models import SRCNN

# Suppress missing triton errors while compiling
torch._dynamo.config.suppress_errors = True


def calc_psnr(MSE):
    return 10 * torch.log10(1 / MSE)


def saveEpochStats(epoch, train_loss, eval_loss, psnr, scale, train_name):
    csv_file = open(
        f"./DLAM_weights/{train_name}_epoch_stats_srcnn_x{scale}.csv", mode="a"
    )
    writer = csv.writer(csv_file)
    writer.writerow([epoch + 1, train_loss, eval_loss, psnr])
    csv_file.close()


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SRCNN()
    # model.load_state_dict(torch.load('./weights/srcnn_x4.pth', map_location=device))
    # model = torch.compile(model)
    print("Using device: ", device)
    model.to(device)

    # G-Loss params
    patch_size = 8
    #gradloss_weight = 0.01

    # Train params
    epochs = 150
    batch_size = 16
    num_workers = 8
    lr = 1e-4
    scale = 3

    # base -> from github repo (just to see if we can reproduce the results) -- PSNR: 33.18
    # g_loss_mse -> with gradient variance loss -- PSNR: 33.10, 32.90
    # g_loss_ssim -> with gradient variance loss and ssim -- PSNR:
    # adam_w -> with adam_w optimizer -- PSNR: 33.23
    # gelu -> with gelu activation -- PSNR: 32.99
    # TODO: change
    train_name = "residuals"

    train_file = f"./91_Image_train/91-image_x{scale}.h5"
    eval_file = f"./Set_5_eval/Set5_x{scale}.h5"

    mse_criterion = nn.MSELoss()
    #grad_criterion = GradientVariance(patch_size=patch_size).to(device)
    #ssim_criterion = SSIM().to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.conv1.parameters()},
            {"params": model.conv2.parameters()},
            {"params": model.conv3.parameters(), "lr": lr * 0.1},
            {"params": model.residual_conv1.parameters()},
            {"params": model.residual_conv2.parameters()},
            {"params": model.residual_conv3.parameters()}
        ],
        lr=lr
    )

    train_dataset = TrainDataset(train_file)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_dataset = EvalDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_psnr = float("-inf")

    for epoch in range(epochs):
        train_loss = 0
        eval_loss = 0
        psnr = 0
        model.train()
        for data in tqdm(train_dataloader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            #loss_ssim = ssim_criterion(preds, labels)

            loss = mse_criterion(preds, labels)

            #loss_grad = gradloss_weight * grad_criterion(preds, labels)

            #loss = loss_mse + loss_grad

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.sum().item()
        train_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            for data in eval_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = mse_criterion(preds, labels)

                # loss_ssim = ssim_criterion(preds, labels)

                #loss_grad = gradloss_weight * grad_criterion(preds, labels)

                #loss = loss_mse + loss_grad

                eval_loss += loss.sum().item()

                psnr += calc_psnr(loss).sum().item()

        eval_loss /= len(eval_dataloader)
        psnr /= len(eval_dataloader)

        if psnr > best_psnr:
            print(f"New Best... saving model with PSNR: {psnr}")
            best_psnr = psnr
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "psnr": psnr,
                },
                f"./DLAM_weights/{train_name}_srcnn_x{scale}.pth",
            )

        saveEpochStats(epoch, train_loss, eval_loss, psnr, scale, train_name)
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Eval Loss: {eval_loss} - PSNR: {psnr}"
        )
