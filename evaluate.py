import paddle

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with paddle.no_grad():
        for x, y in test_loader:
            logits = model(x)
            pred = paddle.argmax(logits, axis=1)
            correct += (pred == y).astype('int').sum().item()
            total += y.shape[0]
    print(f"Validation Accuracy: {correct / total:.4f}")
