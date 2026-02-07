import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
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
    return acc, p, r, f1


def train(model, train_dl, val_dl, args, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    wait = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_dl, loss_fn, optimizer, device
        )
        val_loss = train_one_epoch(
            model, val_dl, loss_fn, optimizer, device
        )

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), "best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping triggered.")
                break
