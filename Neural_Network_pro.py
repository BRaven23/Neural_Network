#!/usr/bin/env python
# coding: utf-8

# # # Neural Network From Scratch
# **Module 11 Project**
# Objective: Build a simple feedforward neural network (one hidden layer) to classify binary 5×6 images of letters **A**, **B**, and **C** using only NumPy. We'll train with backpropagation and visualize results.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # ## 1. Create binary 5x6 patterns for letters A, B, C
# We'll define each letter as a 5x6 binary grid (5 columns × 6 rows = 30 pixels).  
# You can change or augment patterns later for more variety / noise.

# In[3]:


def pattern_to_vector(pat):
    arr = np.array(pat, dtype=float)
    return arr.flatten()  # length 30

# Letter A (stylized)
A = [
    [0,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,1,1,1,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
]

# Letter B
B = [
    [1,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,1,1,1,0],
    [1,0,0,0,1],
    [1,1,1,1,0],
]

# Letter C
C = [
    [0,1,1,1,1],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [1,0,0,0,0],
    [0,1,1,1,1],
]

vecA = pattern_to_vector(A)
vecB = pattern_to_vector(B)
vecC = pattern_to_vector(C)

# Stack into dataset
X_base = np.vstack([vecA, vecB, vecC])  # shape (3, 30)
y_base = np.array([0, 1, 2])            # class indices for A,B,C

print("Shapes:", X_base.shape, y_base.shape)


# # ## 2. (Optional) Augment data a little by adding noisy variations
# Having more training samples helps. We'll create small noisy variants by flipping a few pixels randomly.

# In[4]:


def augment_data(X, y, n_variants=50, noise_rate=0.05, random_seed=1):
    rng = np.random.default_rng(random_seed)
    X_aug = []
    y_aug = []
    for xi, yi in zip(X, y):
        X_aug.append(xi)
        y_aug.append(yi)
        for _ in range(n_variants):
            noisy = xi.copy()
            flips = rng.random(noisy.shape) < noise_rate
            noisy[flips] = 1 - noisy[flips] 
            X_aug.append(noisy)
            y_aug.append(yi)
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    return X_aug, y_aug

X, y = augment_data(X_base, y_base, n_variants=100, noise_rate=0.06, random_seed=42)
print("Augmented shapes:", X.shape, y.shape)


# In[6]:


def one_hot(labels, num_classes):
    oh = np.zeros((labels.size, num_classes))
    oh[np.arange(labels.size), labels] = 1
    return oh

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_x):
    return sigmoid_x * (1 - sigmoid_x)

def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(probs, targets_onehot):
    N = probs.shape[0]
    clipped = np.clip(probs, 1e-12, 1.0)
    return -np.sum(targets_onehot * np.log(clipped)) / N


# In[7]:


# Cell: Neural network class (NumPy-based)
class TwoLayerNN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.5, seed=1):
        rng = np.random.default_rng(seed)
        # Xavier-ish init
        self.W1 = rng.normal(0, np.sqrt(2 / (input_dim + hidden_dim)), size=(input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, np.sqrt(2 / (hidden_dim + output_dim)), size=(hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def forward(self, X):
        # X shape: (N, D)
        z1 = X @ self.W1 + self.b1  # (N, H)
        a1 = sigmoid(z1)            # (N, H)
        z2 = a1 @ self.W2 + self.b2 # (N, C)
        probs = softmax(z2)         # (N, C)
        cache = (X, z1, a1, z2, probs)
        return probs, cache

    def backward(self, cache, targets_onehot):
        X, z1, a1, z2, probs = cache
        N = X.shape[0]

        # Output layer gradient (softmax + cross-entropy simplifies)
        dZ2 = (probs - targets_onehot) / N   # (N, C)
        dW2 = a1.T @ dZ2                     # (H, C)
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1, C)

        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T                # (N, H)
        dZ1 = dA1 * sigmoid_derivative(a1)   # (N, H)
        dW1 = X.T @ dZ1                      # (D, H)
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, H)

        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)


# In[8]:


# Cell: Training loop
np.random.seed(0)
input_dim = X.shape[1]    # 30
hidden_dim = 20           # tuneable
output_dim = 3            # A, B, C

model = TwoLayerNN(input_dim, hidden_dim, output_dim, lr=1.0, seed=2)

# Convert labels to one-hot
Y_onehot = one_hot(y, output_dim)

epochs = 200
loss_history = []
acc_history = []

for epoch in range(1, epochs + 1):
    probs, cache = model.forward(X)
    loss = cross_entropy_loss(probs, Y_onehot)
    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y)

    loss_history.append(loss)
    acc_history.append(acc)

    model.backward(cache, Y_onehot)

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {acc:.4f}")

# Final metrics
probs_final, _ = model.forward(X)
print("\nFinal training accuracy:", np.mean(np.argmax(probs_final, axis=1) == y))


# In[9]:


# Cell: Plots
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(acc_history, label='Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()


# In[10]:


# Cell: Test on base patterns
X_test = X_base.copy()
y_test = y_base.copy()
probs_test, _ = model.forward(X_test)
preds_test = np.argmax(probs_test, axis=1)

print("True labels:", y_test)
print("Predicted:", preds_test)
print("Predicted probs:\n", probs_test)

# Visualize each image
fig, axes = plt.subplots(1, 3, figsize=(9,3))
titles = ['A', 'B', 'C']
for i, ax in enumerate(axes):
    img = X_test[i].reshape(6,5)  # reshape back to 6 x 5 (rows x cols)
    ax.imshow(img, cmap='gray_r', interpolation='nearest')
    ax.set_title(f"True: {titles[i]} | Pred: {titles[preds_test[i]]}")
    ax.axis('off')
plt.show()


# In[11]:


np.savez("nn_weights.npz", W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
print("Saved weights to nn_weights.npz")


# In[ ]:




