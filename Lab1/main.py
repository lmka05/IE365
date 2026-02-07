import argparse
import torch
from dataset import get_dataloaders
from model import build_model
from utils import train, evaluate, get_loss_fn, get_optimizer, plot_training_curves



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/kaggle/working/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--optimizer",choices=['adam', 'sgd'],default='adam')

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, test_dl = get_dataloaders(
        args.data_dir,
        args.img_size,
        args.batch_size,
        args.augment
    )

    model = build_model(args).to(device)
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(model, args)

    train_loss_list, val_loss_list, val_acc_list = train(model, train_dl, val_dl, loss_fn, optimizer, args, device)
    plot_training_curves(train_loss_list, val_loss_list, val_acc_list)

    acc, p, r, f1 = evaluate(model, test_dl, device)
    print(f"Test results on test set: Acc={acc:.4f}, 'Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")


if __name__ == "__main__":
    main()
