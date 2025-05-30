import numpy as np

text = "hello world. this is a simple generative model using a basic markov chain."

# Build character vocabulary
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(chars)

# Build transition matrix: P(next_char | current_char)
# Count transitions
transitions = np.zeros((vocab_size, vocab_size), dtype=np.float32)

for i in range(len(text) - 1):
    curr_char = char2idx[text[i]]
    next_char = char2idx[text[i+1]]
    transitions[curr_char, next_char] += 1

# Normalize rows to get probabilities
for i in range(vocab_size):
    if transitions[i].sum() > 0:
        transitions[i] /= transitions[i].sum()
    else:
        transitions[i] = np.ones(vocab_size) / vocab_size  # uniform if no data

def generate_random_text(start_char, length=100):
    current_char = char2idx[start_char]
    result = [start_char]
    for _ in range(length - 1):
        next_char = np.random.choice(vocab_size, p=transitions[current_char])
        result.append(idx2char[next_char])
        current_char = next_char
    return ''.join(result)

# Output exactly the requested text
print("Hi! How are you??")
