import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

def get_optimizer(model, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
def get_loss_fn():
    return torch.nn.CrossEntropyLoss()

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    return acc, p, r, f1, y_true, y_pred


def train(model, train_dl, val_dl, loss_fn, optimizer, args, device):
    best_val_loss = float("inf")
    wait = 0
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_dl, loss_fn, optimizer, device
        )
        val_loss = train_one_epoch(
            model, val_dl, loss_fn, optimizer, device
        )
        acc, _, _, _, _, _= evaluate(model, val_dl, device)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), "best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping triggered.")
                break
        return train_loss_list, val_loss_list, val_acc_list

def plot_training_curves(train_loss_list, val_loss_list, val_acc_list):
    epochs = range(1, len(train_loss_list) + 1)
    fig, axes = plt.subplots(1,2,figsize=(12, 5))
    axes =axes.flatten()

    axes[0].plot(epochs, train_loss_list, label="Train Loss")
    axes[0].plot(epochs, val_loss_list, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, val_acc_list, label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    

