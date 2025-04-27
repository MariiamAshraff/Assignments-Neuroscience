import numpy as np

vocab = ['I', 'love', 'deep', 'learning']
word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}
vocab_size = len(vocab)
hidden_size = 5
learning_rate = 0.1

def one_hot(idx, size):
    vec = np.zeros((size, 1))
    vec[idx] = 1
    return vec

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x) ** 2
np.random.seed(0)
W_x = np.random.randn(hidden_size, vocab_size) * 0.1
W_h = np.random.randn(hidden_size, hidden_size) * 0.1
W_y = np.random.randn(vocab_size, hidden_size) * 0.1

for epoch in range(100):
    inputs = ["I", "love", "deep"]
    target = "learning"
    xs, hs = {}, {}
    hs[-1] = np.zeros((hidden_size, 1))

    for t, word in enumerate(inputs):
        xs[t] = one_hot(word_to_ix[word], vocab_size)
        hs[t] = tanh(np.dot(W_x, xs[t]) + np.dot(W_h, hs[t - 1]))

    y = softmax(np.dot(W_y, hs[len(inputs) - 1]))
    target_vec = one_hot(word_to_ix[target], vocab_size)
    loss = -np.sum(target_vec * np.log(y))

    dW_x = np.zeros_like(W_x)
    dW_h = np.zeros_like(W_h)
    dW_y = np.zeros_like(W_y)
    dh_next = np.zeros_like(hs[0])

    dy = y - target_vec
    dW_y += np.dot(dy, hs[len(inputs) - 1].T)

    for t in reversed(range(len(inputs))):
        dh = np.dot(W_y.T, dy) + dh_next
        dh_raw = dh * dtanh(np.dot(W_x, xs[t]) + np.dot(W_h, hs[t - 1]))
        dW_x += np.dot(dh_raw, xs[t].T)
        dW_h += np.dot(dh_raw, hs[t - 1].T)
        dh_next = np.dot(W_h.T, dh_raw)

    for param, dparam in zip([W_x, W_h, W_y], [dW_x, dW_h, dW_y]):
        param -= learning_rate * dparam

    if epoch % 10 == 0:
        pred_word = ix_to_word[np.argmax(y)]
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Predicted: {pred_word}")
        