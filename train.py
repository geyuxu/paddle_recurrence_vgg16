import argparse
from model import VGG16
from dataset import get_dataset, get_transforms
from utils import save_model
import paddle
import paddle.nn as nn

def train(args):
    transform = get_transforms((args.input_size, args.input_size))
    train_dataset = get_dataset(args.dataset, 'train', transform)
    test_dataset = get_dataset(args.dataset, 'test', transform)
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=args.batch_size)

    model = VGG16(num_classes=args.num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_id, (x, y) in enumerate(train_loader):
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_id % 100 == 0:
                print(f"[Epoch {epoch}] Batch {batch_id}, Loss: {loss.numpy():.4f}")
        save_model(model, optimizer, epoch, args.save_dir)

        from evaluate import evaluate
        evaluate(model, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    train(args)
