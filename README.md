# GPT-2 Transformer Model in TensorFlow

## Overview
This repository contains an implementation of a GPT-2 (Generative Pre-trained Transformer) model using TensorFlow and Keras. The model is built from scratch using transformer blocks, multi-head self-attention, and feed-forward networks. It can be used for text generation, natural language processing tasks, and fine-tuning on custom datasets.

## Features
- Implements **Multi-Head Self-Attention** for contextual understanding.
- Uses **Feed-Forward Networks** with GELU activation.
- Includes **Positional Embeddings** to retain sequence information.
- Supports **Custom Vocabulary Size and Sequence Length**.
- Optimized for training using **Layer Normalization and Dropout**.

## Installation
To run this model, install the required dependencies:
```bash
pip install tensorflow
```

## Model Architecture
The model consists of the following components:

### 1. Multi-Head Self-Attention (MHSA)
- Splits input into multiple attention heads.
- Computes attention scores and applies softmax.
- Projects the output back to the embedding dimension.

### 2. Feed-Forward Network (FFN)
- Two fully connected layers with a GELU activation.
- Helps the model learn complex representations.

### 3. Transformer Block
- Combines MHSA and FFN with residual connections and layer normalization.
- Uses dropout for regularization.

### 4. GPT-2 Model
- Embedding layer for tokens and positions.
- Stacks multiple transformer blocks.
- Outputs logits for each token in the vocabulary.

## Usage
```python
import tensorflow as tf
from model import GPT2

VOCAB_SIZE = 50257
MAX_LENGTH = 1024

gpt2 = GPT2(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH)
inputs = tf.keras.Input(shape=(MAX_LENGTH,), dtype=tf.int32)
outputs = gpt2(inputs)
model = tf.keras.Model(inputs, outputs)
model.summary()
```

## Model Summary
```bash
Model: "gpt2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Embedding (None, MAX_LENGTH, 768)  38598144   
TransformerBlock x 12        (None, MAX_LENGTH, 768)  110620416  
LayerNormalization           (None, MAX_LENGTH, 768)  1536      
Dense (Output)               (None, MAX_LENGTH, VOCAB_SIZE)  38619840  
=================================================================
```


## Author
Developed by [ABILASH G].

## Contributing
Feel free to contribute by submitting pull requests or opening issues!

