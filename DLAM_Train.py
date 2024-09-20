import csv
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets import EvalDataset, TrainDataset
from models import SRCNN

def calc_psnr(mse):
    return 10 * torch.log10(1 / mse)

def save_epoch_stats(epoch, train_loss, eval_loss, psnr, scale, train_name):
    csv_file = open(
        f"./weights/{train_name}_epoch_stats_srcnn_x{scale}.csv", mode="a"
    )
    writer = csv.writer(csv_file)
    writer.writerow([epoch + 1, train_loss, eval_loss, psnr])
    csv_file.close()

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SRCNN(num_channels=1) # 1 Grayscale, 3 RGB
    print("Using device:", device)
    model.to(device)

    epochs = 150
    batch_size = 128
    num_workers = 8
    scale = 3

    lr = 1e-4
    lr_last_layer = 1e-5

    train_file = f"./91_Image_train/91-image_x{scale}.h5"
    eval_file = f"./Set_5_eval/Set5_x{scale}.h5"

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(
        [
            {"params": model.conv1.parameters(), "lr": lr},
            {"params": model.conv2.parameters(), "lr": lr},
            {"params": model.conv3.parameters(), "lr": lr_last_layer}
        ]
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
        model.train()
        train_loss = 0.0

        for data in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataloader.dataset)

        model.eval()
        eval_loss = 0.0
        psnr_total = 0.0

        with torch.no_grad():
            for data in eval_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)
                eval_loss += loss.item()

                psnr = calc_psnr(loss.item())
                psnr_total += psnr

        eval_loss /= len(eval_dataloader)
        psnr_avg = psnr_total / len(eval_dataloader)

        if psnr_avg > best_psnr:
            print(f"New Best PSNR: {psnr_avg:.4f} dB - Saving model")
            best_psnr = psnr_avg
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "psnr": psnr_avg,
                },
                f"./weights/srcnn_x{scale}.pth",
            )

        save_epoch_stats(epoch, train_loss, eval_loss, psnr_avg, scale, "srcnn")

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.6f} - Eval Loss: {eval_loss:.6f} - PSNR: {psnr_avg:.2f} dB"
        )
