import numpy as np
import matplotlib.pyplot as plt
from nanograd.tensor import Tensor
from nanograd.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from nanograd.optim import Adam
from nanograd.data import load_mnist, get_batches

# hyperparameters
EPOCHS     = 20
BATCH_SIZE = 32
LR         = 0.001

# data
X_train, y_train, X_test, y_test = load_mnist()

# model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10)
)

# loss + optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

# tracking
train_losses = []
test_accuracies = []

def accuracy(X, y):
    correct = 0
    for i in range(0, len(X), BATCH_SIZE):
        xb = Tensor(X[i:i+BATCH_SIZE], requires_grad=False)
        logits = model(xb)
        preds = np.argmax(logits.data, axis=1)
        correct += (preds == y[i:i+BATCH_SIZE]).sum()
    return correct / len(X)

# training loop
for epoch in range(EPOCHS):
    epoch_loss = 0
    num_batches = 0

    for xb, yb in get_batches(X_train, y_train, BATCH_SIZE):
        # 1. forward
        xb = Tensor(xb, requires_grad=False)
        logits = model(xb)

        # 2. loss
        loss = criterion(logits, yb)

        # 3. backward
        loss.backward()

        # 4. update weights
        optimizer.step()

        # 5. zero grads
        optimizer.zero_grad()

        epoch_loss += loss.data
        num_batches += 1

    # epoch stats
    avg_loss = epoch_loss / num_batches
    acc = accuracy(X_test, y_test)
    train_losses.append(avg_loss)
    test_accuracies.append(acc)

    print(f'epoch {epoch+1}/{EPOCHS} | loss: {avg_loss:.4f} | test acc: {acc*100:.2f}%')

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(test_accuracies)
ax2.set_title('Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

print(f'final test accuracy: {test_accuracies[-1]*100:.2f}%')