import numpy as np

print("=== N=8 PARITY: FULL ENUMERATION SANITY CHECK ===\n")

# Generate ALL 2^8 = 256 possible bit sequences
n = 8
all_inputs = []
all_labels = []

for i in range(2**n):
    bits = np.array([(i >> j) & 1 for j in range(n)])  # binary representation
    x = 2 * bits - 1  # encode as -1/+1
    parity = bits.sum() % 2  # 0 or 1
    all_inputs.append(x)
    all_labels.append(parity)

X = np.array(all_inputs, dtype=np.float32)
y = np.array(all_labels, dtype=np.float32)

print(f"Full enumeration: {len(X)} inputs")
print(f"Label balance: {y.mean():.2f} (should be 0.5)")

# MLP with BCE loss (proper binary classification)
class MLP_BCE:
    def __init__(self, input_dim, hidden_dim=32):
        # Xavier init
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(1, dtype=np.float32)

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU
        self.logits = (self.a2 @ self.W3 + self.b3).squeeze()
        return self.logits

    def backward(self, x, y, lr=0.1):
        batch_size = x.shape[0]
        # BCE gradient: sigmoid(logit) - y
        probs = 1 / (1 + np.exp(-np.clip(self.logits, -500, 500)))
        dlogits = (probs - y).reshape(-1, 1) / batch_size

        dW3 = self.a2.T @ dlogits
        db3 = dlogits.sum(axis=0)
        da2 = dlogits @ self.W3.T
        dz2 = da2 * (self.z2 > 0)

        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)

        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

def bce_loss(logits, y):
    logits = np.clip(logits, -500, 500)
    return np.mean(np.maximum(logits, 0) - logits * y + np.log(1 + np.exp(-np.abs(logits))))

def accuracy(logits, y):
    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
    preds = (probs > 0.5).astype(int)
    return (preds == y).mean()

# TEST A: Train on ALL 256 (train == test)
print("\n--- TEST A: Train on ALL 256 (wd=0, BCE loss) ---")
np.random.seed(42)
mlp = MLP_BCE(n, hidden_dim=64)

for epoch in range(1, 5001):
    logits = mlp.forward(X)
    mlp.backward(X, y, lr=0.1)

    if epoch % 500 == 0 or epoch <= 10:
        loss = bce_loss(logits, y)
        acc = accuracy(logits, y)
        print(f"Epoch {epoch:4d} | loss={loss:.4f} | acc={acc:.1%}")
        if acc >= 0.99:
            print(f">>> CONVERGED at epoch {epoch}")
            break

final_acc = accuracy(mlp.forward(X), y)
print(f"\nFINAL (train-all): {final_acc:.1%}")
if final_acc > 0.95:
    print("SUCCESS: MLP can represent parity function")
else:
    print("FAILURE: MLP cannot fit even the full truth table")

# TEST B: 80/20 split
print("\n--- TEST B: 80/20 split ---")
np.random.seed(42)
perm = np.random.permutation(256)
split = int(0.8 * 256)
X_train, y_train = X[perm[:split]], y[perm[:split]]
X_test, y_test = X[perm[split:]], y[perm[split:]]

mlp2 = MLP_BCE(n, hidden_dim=64)
for epoch in range(1, 10001):
    logits = mlp2.forward(X_train)
    mlp2.backward(X_train, y_train, lr=0.1)

    if epoch % 1000 == 0:
        train_acc = accuracy(logits, y_train)
        test_acc = accuracy(mlp2.forward(X_test), y_test)
        print(f"Epoch {epoch:4d} | train={train_acc:.1%} | test={test_acc:.1%}")

print(f"\nFINAL 80/20: train={accuracy(mlp2.forward(X_train), y_train):.1%}, test={accuracy(mlp2.forward(X_test), y_test):.1%}")
