<!-- markdownlint-disable MD024 MD025 -->

# **Lecture 1,2 - Tips & Tricks**

## Working with Pre-trained Model

### Finetuning

> You can do Transfer Learning first and then after training new layers do full Finetuning

### Transfer Learning

### LoRA

[PERF](https://github.com/huggingface/peft) - HuggingFace's LoRA

### Distillation

### Quantizattion

### Triplet loss

## Convergence

### Learning Rate Scheduler

Adam's adaptiveness is not enough.

`torch.optim.lr_scheduler.ReduceLROnPlateau` is nice!

![alt text](notes_images/learning_rate_schedulers.png)

### Warmup

_Statistics gathering_ optimizers need a bit of first steps in order to see path way clearly.

![alt text](notes_images/learning_rate_warmup.png)

### Loss Scaling

![alt text](notes_images/loss_scaling.png)

## Overfitting

### Label Smoothing

Changing one-hot to weak one-hot:

[0, 0, 1, 0, 0, 0] $\to$ [0.02, 0.02, 0.9, 0.02, 0.02, 0.02]

> Not that often used.

### Temperature

Use this like with Learning Rate Scheduler (inversed)!

### Noise

Adding noise to input data (especially good with images).

### Augmentations

> Don't do augmentations in validation/test sets.

# **Seminar 1**

## Musthaves

- Do BatchNorm _before_ DropOut.

- Paste only parameters that `require_grad` into optimizer.

- Via increasing hidden layer size (10 -> 20 -> 30 -> ...) we try to learn heavier (larger) patterns in our data.

## Training & Evaluation

```py
criterion = nn.NLLLoss()

def train(model, optimizer, loader, criterion):
    model.train()
    losses_tr = []

    for images, targets in tqdm(loader):
        # Zero out gradients
        optimizer.zero_grad()

        # Move everything to the correct device
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass and loss computation
        out = model(images)
        loss = criterion(out, targets)

        # Backward pass, optimizer step
        loss.backward()
        optimizer.step()

        losses_tr.append(loss.item())

    return model, optimizer, np.mean(losses_tr)


def val(model, loader, criterion, metric_names=None):
    model.eval()
    losses_val = []

    if metric_names:
        metrics = {name: [] for name in metric_names}

    with torch.no_grad():
        for images, targets in tqdm(loader):
            # Same as training, but without backward pass and optimizer step
            images = images.to(device)
            targets = targets.to(device)
            out = model(images)
            loss = criterion(out, targets)
            losses_val.append(loss.item())

            if metric_names and 'accuracy' in metrics:
                _, pred_classes = torch.max(out, dim=1)
                metrics['accuracy'].append(
                    (pred_classes == targets).float().mean().item()
                )

    if metric_names:
        for name in metrics:
            metrics[name] = np.mean(metrics[name])

    if metric_names:
        return np.mean(losses_val), metrics
    else:
        return None
```

### Learning Loop

```py
def learning_loop(
    model,
    optimizer,
    train_loader,
    val_loader,
    criterion,
    scheduler=None,
    min_lr=None,
    epochs=10,
    val_every=1,
    draw_every=1,
    metric_names=None
):
    losses = {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        print(f'{epoch}/{epochs}:')

        model, optimizer, loss = train(model, optimizer, train_loader, criterion)
        losses['train'].append(loss)

        if not (epoch % val_every):
            pass
            # `scheduler.step()` here too

        if not (epoch % draw_every):
            pass
```

### `model.train()` & `model.eval()`

| Mode          | Dropout                        | BatchNorm                                |
|---------------|--------------------------------|------------------------------------------|
| `model.train()` | ON (randomly drops neurons)    | Uses current batch’s mean & variance      |
| `model.eval()`  | OFF (all neurons active)       | Uses running averages collected in training |

# **Lecture 2 - Attention & Transormers**

## Embeddings

1. Word embeddings
2. Text embeddings (sentence)

## Sequence Tasks

### Seq2One

![alt text](notes_images/seq2one.png)

### One2Seq

![alt text](notes_images/one2seq.png)

### Seq2Seq Same

![alt text](notes_images/seq2seq_same.png)

### Seq2Seq Different

![alt text](notes_images/seq2seq_diff.png)

## Transformer

[Attention Is All You Need [2017]](https://arxiv.org/abs/1706.03762)

## Attention (Idealogically)

All RNN-like architectures suffer from forgetting information.

- Relevancy **score**: $ s_i = score(h_i, z) $ - can be scalar multiplication, bilinear form, NN, ... everything that can turn $h_i$ and $z$ into single number.

- Probabilities: $ a_1, a_2, \ldots = \mathrm{softmax}(s_1, s_2, \ldots) $

- Context: $ c = \sum_i a_i h_i $

- New feature set: $ \tilde{z} = [c; z] $

![alt text](notes_images/attention_idea.png)

## Multi-Head Attention

### Step 1: Input Representations

$$
X \in \mathbb{R}^{n_{sequence} \times d_{\text{embedding}}}
$$
where row $X_i$ is the embedding of token $i$.  

### Step 2: Q, K, V

The Transformer learns **three linear projection matrices per head**:

$$
W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

Then:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

- $Q \in \mathbb{R}^{n \times d_k}$ (queries) - "What am I looking for?"

- $K \in \mathbb{R}^{n \times d_k}$ (keys) - "What properties do I have?"

- $V \in \mathbb{R}^{n \times d_v}$ (values) - "What information should I pass on if selected?"

### Step 3: Scoring Queries Against Keys

How much should token $i$ pay attention to token $j$?

The raw **attention score** between query $Q_i$ and key $K_j$ is:

$$
\text{score}(i,j) = Q_i \cdot K_j^T = \sum_{m=1}^{d_k} Q_{i,m} K_{j,m}
$$

$$
\text{Scores} = Q K^T \in \mathbb{R}^{n \times n}
$$

Row $i$ = how much token $i$ attends to all tokens.

### Step 4: Scaling by $\sqrt{d_k}$

#### **Assumptions**

1. $$
    \mathbb{E}[k_i] = \mathbb{E}[q_i] = 0, \quad \mathrm{Var}[k_i] = \mathrm{Var}[q_i] = 1
    $$

    [Under standard weight initialization (e.g., Xavier) that is okay :)]

2. $ k_i, q_i $ are _independent_ across dimensions

Then:

$$
\mathrm{Var}[K^\top Q]
= \sum_{i=1}^{d_k} \mathrm{Var}[k_i q_i]
= \sum_{i=1}^{d_k} \mathrm{Var}[k_i] \mathrm{Var}[q_i]
= d_k
$$

#### **Scaling**

Because large values push softmax into saturation, leading to vanishing gradients, we scale by $\sqrt{d_k}$ (like temperature):

$$
\text{ScaledScores} = \frac{Q K^T}{\sqrt{d_k}}
$$

### Step 5: Attention

**Attention weights**:

$$
A = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}
$$

- Row $i$: distribution over which tokens $i$ attends to.  
- Each row sums to 1.  

**Attention output**:

$$
\text{Attention}(Q, K, V) = AV
$$

### Step 6: Multi-Head Attention

> _multi_-head attention instead of a _single_ head to focus on different representation subspaces: Syntactic structure, Positional patterns, Semantic roles, ...

For $h$ heads, we repeat the above with different projection matrices:

$$
\text{head}_i = \text{Attention}(X W_i^Q, \, X W_i^K, \, X W_i^V)
$$

Then concatenate:

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

with $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$.

## Positional Encoding

Since the Transformer has no recurrence/convolution, it adds positional encodings (sinusoidal or learned) to capture order.

The **position encoding** is just _added_ to embedding.

### Sinusoidal Positional Encoding (from the paper)

Vaswani et al. (2017) proposed **fixed sinusoidal encodings**.

For a token at position $pos$ and dimension index $i$:

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

- $pos \in [0, n-1]$ = token position in the sequence.  
- $i \in [0, d_{\text{model}}/2 - 1]$.  
- $d_{\text{model}}$ = embedding dimension.  

### Learnable Positional Encoding

An alternative: treat position embeddings as **trainable parameters**:

$$
PE \in \mathbb{R}^{n_{\text{max}} \times d_{\text{model}}}
$$

- Advantage: more flexibility.  
- Disadvantage: poor extrapolation to longer sequences.

### Relative Positional Encoding (Modern Variant)

Later work (Transformer-XL, T5, GPT-NeoX) often uses **relative encodings**:

- Instead of absolute positions, attention scores depend on _relative distance_ between tokens.  

$$
\text{score}(i,j) = \frac{Q_i K_j^T + Q_i R_{i-j}^T}{\sqrt{d_k}}
$$

where $R_{i-j}$ encodes the relative offset.  

This improves _long_-context handling.

## Encoder-Decoder

![alt text](notes_images/transformer.png)

- **Encoder** (left block): processes the input sequence into a contextual representation.

- **Decoder** (right block): generates the output sequence one token at a time.

  Attends both to previously generated tokens and the encoder’s representation:

  - Queries $\{Q_i\}$ come from the decoder.
  - Keys $\{K_i\}$ & Values $\{V_i\}$ come from encoder output.

  Allows decoder to focus on relevant input parts when generating output.

> Embeddings flow from the last layer of Encoder block into all Decoder block's attention (not from the respective (соответствующих) layers).

### Masked Multi-Head Attention

The mask is just 0&1 matrix for decoder not to look into future.

## BERT (encoder-only)

Bidirectional Encoder Representations from Transformers.

### Mask Language Modelling

- Randomly mask ~15% of tokens in input to predict.

That's why it's encoder-only, it get's full sequence, but some word are marked ("cat ___ fish") $\to$ learns to fill in.  
So it's not best for predicting the following word, rather having a context fill skipped word(s).

### RoBERTa

| Change             | Description                            |
| ------------------ | -------------------------------------- |
| Removed **NSP**      | Only **MLM** objective used                |
| More data       | Used 160 GB of text (vs BERT’s ~16 GB) |
| Longer training  | Trained for more steps                 |
| **Dynamic masking** | Recomputed masked positions each epoch |
| Larger batches  | Allowed more stable optimization       |

## GPT-x (decoder-only)

Generative Pretrained Transformer.

## BART

Encoder reads corrupted text in full (like BERT).

Decoder generates clean text autoregressively (like GPT).

### Why GPT-x, but not BART?

1. Masked LM (BERT) and denoising (BART) objectives don’t scale as smoothly $\to$ GPT-x fits the “**just throw more data + compute**” recipe better.

2. BART’s denoising autoencoder focuses more on _reconstruction_: summarization, translation, etc., but less natural for open-ended dialogue, code writing, reasoning $\to$ GPT objective = more general-purpose.

# **Lecture 3 - DL Interpretation**

## _CV part_

## 1. Receptive Field

**Receptive field**: for a given neuron in that feature map, it’s the region of the original input image that could affect that neuron’s value.

![alt text](notes_images/convolutions.png)

## 2. Deconvolutional Network

0. Start with some feature map (CNN's layer) [usually, last]

1. **Deconvolution** - `conv2d_transpose` - apply the transpose of the convolution filter to map activations back toward pixel space.

2. **Unpolling** - reverse pooling by placing activations back into their original pooled positions.

3. **Rectification** - apply the same nonlinearity rules (e.g., keep only positive signals).

4. Repeat layer by layer until reaching the input.

![alt text](notes_images/deconv1.png)

![alt text](notes_images/deconv2.png)

## 3. Gradient-Based

- **Saliency maps** = interpret existing input:

  - Freeze image.
  - Compute gradients of output w.r.t. input.
  - Highlight which input pixels matter for the current prediction.

- **Activation maximization** = generate prototype input.

  - Freeze model parameters.
  - Start with a random/noise image.
  - Optimize the input image itself via gradient ascent so that it maximally activates a chosen neuron/class.

  > Kinda looks like picture for adversarial attack!

  The result is a synthetic image showing what the model “thinks” that class looks like.

  ![alt text](notes_images/gradient_based.png)

  $\to$ CNN's look at textures, not at objects.

## 4. Guided Backpropagation

**Guided Backpropagation** improves saliency maps by modifying how ReLUs are handled during backprop.

- In normal backprop:  
    If the forward ReLU output was zero, its gradient is blocked.

- In guided backprop:  

    A gradient is passed only if both the _forward activation and the backward gradient_ (`deconv`) are positive (where it's negative $\to$ 0).

    This “guides” the signal to highlight pixels that positively contribute to the activation.

![alt text](notes_images/guided_backprop1.png)

![alt text](notes_images/guided_backprop2.png)

## 5. CAM

### GAP

**GAP (Global Average Pooling)** compresses each feature map into one value by averaging all its activations, replacing dense layers at the end of CNNs.

So if the last conv layer has 512 feature maps, GAP produces a 512-dimensional vector.

**CAM (Class Activation Mapping)** only works with architectures that end with GAP before the final classification layer.

The _heatmap_ as a weighted sum of feature maps is produced.

## 6. GradCAM

- CAM: directly uses classifier weights.

- Grad-CAM: uses gradients to infer “which feature maps matter” for the class of interest.

## 7. Guided GradCAM

![alt text](notes_images/gradcam.png)

## _LLM part_

Transformer is super easy interpreted, because attention gives importance of each (outputed) word to each (inputted) word.

# **Seminar 3**

## Working with Parameters

`nn.Parameter` - learnable parameters.

```py
class MyNNLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return x @ self.weights + self.bias


class MyNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l0 = MyNNLinear(input_size, 2)
        self.l1 = MyNNLinear(2, output_size)
    
    def forward(self, x):
        x = self.l0(x)
        x = F.relu(x)
        x = self.l1(x)
        return x


class MyBiggerNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net0 = MyNet(input_size, 3)
        self.net1 = MyNet(3, output_size)
    
    def forward(self, x):
        x = self.net0(x)
        x = F.relu(x)
        x = self.net1(x)
        return x

net = MyBiggerNet(4, 1)
print([*net.named_children()])
print([*net.net0.l0.parameters()])
```

```py
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    "loss": loss,
}, path)

# in reality we could save more frequently, because for LLM's epoch could be 10.000 batches or more.
```

### `torch.register_buffer`

Buffer are non-trainable parameters.

For example, `running_mean`, `running_var` statistics in BatchNorm are buffers.

## Xavier Initialization

[2010, Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

The goal is to keep the variance of activations (forward pass) and the variance of gradients (backward pass) roughly the same across all layers.  
If the variance shrinks or grows layer by layer, you get _vanishing_ or _exploding gradients_.

> Best suited for **sigmoid** or **tanh** activations.

### Idea

For a neuron: $ y = \sum_{i=1}^{n_{in}} W_i x_i $

- Step 1. Forward pass (activations)

    $$
    Var(y) = n_{in} \cdot Var(W) \cdot Var(x)
    $$

    To keep activations stable:

    $$
    Var(y) \approx Var(x) \quad \Rightarrow \quad Var(W) \approx \frac{1}{n_{in}}
    $$

- Step 2. Backward pass (gradients)

    To keep gradients stable:

    $$
    Var(W) \approx \frac{1}{n_{out}}
    $$

- Step 3. Balance both

    To satisfy both forward and backward constraints, Xavier takes the average:

    $$
    Var(W) = \frac{1}{2}\left(\frac{1}{n_{in}} + \frac{1}{n_{out}}\right)
    = \frac{2}{n_{in} + n_{out}}
    $$

### Formula

$$
Var(W) = \frac{2}{n_{in} + n_{out}}
$$

- **Uniform distribution**:  
    $$
    W \sim U\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
    $$

- **Normal distribution**:  
    $$
    W \sim \mathcal{N}\left(0, \frac{2}{n_{in} + n_{out}}\right)
    $$

## He (Kaiming) initialization

It focuses purely on forward variance preservation, because for `ReLU` the main danger is **vanishing activations**.

> nn.Linear is initialized by He by default!

It ignores the gradient balancing part of Xavier:

$$
Var(W) = \frac{2}{n_{in}}
$$

# **Seminar 3.2**

> HuggingFace's website is not up to date $\to$ better to look at HuggingFace's github.

## [Streamlit](https://github.com/streamlit/streamlit)

```py
pip install streamlit
```

# **Lecture 4 - Generative NN**

## 1. Comparing Distributions

Task: make the **model distribution** $ p_\theta(x) $ approximate the **true data distribution** $ p_{\text{data}}(x) $.

- How do we measure the “distance” between two distributions?

### 1.1 Total Variation Distance (TVD)

$$
\mathrm{TVD}(p, q) = \frac{1}{2} \int_{\mathcal{X}} |p(x) - q(x)| \, dx
$$

- Symmetric, bounded in $[0,1]$.  
- Not smooth $\to$ problematic for gradient-based optimization when supports are _disjoint_ (too far away).

### 1.2 Kullback–Leibler Divergence (KL)

$$
D_{\mathrm{KL}}(p \,||\, q) = \int_{\mathcal{X}} p(x) \log \frac{p(x)}{q(x)} \, dx
$$

Not symmetric: $ D_{\mathrm{KL}}(p||q) \neq D_{\mathrm{KL}}(q||p) $  

- $ D_{\mathrm{KL}}(p||q) $ penalizes missing support $\to$ mode covering.

    ![alt text](notes_images/kl_forward.png)

- $ D_{\mathrm{KL}}(q||p) $ penalizes assigning mass to unsupported regions $\to$ mode seeking.

    ![alt text](notes_images/kl_reversed.png)

Used in maximum likelihood training (e.g., VAEs via ELBO).

### 1.3 Jensen–Shannon Divergence (JS)

Let $ m(x) = \frac{1}{2}(p(x) + q(x)) $. Then

$$
D_{\mathrm{JS}}(p || q) = \frac{1}{2} D_{\mathrm{KL}}(p || m) + \frac{1}{2} D_{\mathrm{KL}}(q || m)
$$

![alt text](notes_images/js.png)

- Symmetric, bounded in $[0, \log 2]$  
- Finite even with disjoint supports  
- Used in **original GAN** training (Goodfellow et al., 2014)

**Problem**: When distributions are on disjoint manifolds, **gradient vanishes**.

### 1.4 Wasserstein Distance (Earth Mover’s Distance)

$$
W(p, q) = \inf_{\gamma \in \Pi(p,q)} \mathbb{E}_{(x,y) \sim \gamma} [ \| x - y \| ]
$$

where $ \Pi(p,q) $ is the set of joint distributions with marginals $ p $ and $ q $.

**Kantorovich–Rubinstein duality:**

$$
W(p, q) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{x \sim q}[f(x)]
$$

- Think of moving "earth" from $ p $ to match $ q $.  
- Finite and smooth even with disjoint supports.  
- Leads to **Wasserstein GAN (WGAN)** with stable training and meaningful gradients.

| Metric | Symmetric | Finite if supports disjoint? | Smooth Gradients? | Common use |
|--------|-----------|-------------------------------|--------------------|------------|
| TVD    | ✅        | ❌                           | ❌                 | Theory     |
| KL     | ❌        | ❌                           | 🚫 (p,q unstable) | MLE, VAEs  |
| JS     | ✅        | ✅                           | 🚫 (vanishes)      | Vanilla GAN |
| Wasserstein | ✅   | ✅                           | ✅ Good            | WGAN |

## 2. Generation Evaluation Metrics

### 2.1 Inception Score (IS)

Given generator samples $ x = G(z) $ and a classifier giving $ p(y|x) $:

$$
\mathrm{IS} = \exp \left( \mathbb{E}_{x \sim p_g} \left[ D_{\mathrm{KL}}(p(y|x) \,\|\, p(y)) \right] \right)
$$

where

$$
p(y) = \int p(y|x) \, p_g(x) \, dx
$$

#### **Intuition:**

- **Sharpness**: $ p(y|x) $ has low entropy for clear, classifiable images.  
- **Diversity**: $ p(y) $ has high entropy across samples.  
- High IS $\to$ sharp & diverse samples.

#### **Limitations:**

- No direct comparison to real data.  
- Depends on pretrained classifier domain.  

### 2.2 Fréchet Inception Distance (FID)

Let real features: $ \{\mu_r, \Sigma_r\} $, generated features: $ \{\mu_g, \Sigma_g\} $ from Inception embeddings.  

$$
\mathrm{FID}(p_r, p_g) = \| \mu_r - \mu_g \|_2^2 + \mathrm{Tr}\left( \Sigma_r + \Sigma_g - 2 (\Sigma_r \Sigma_g)^{1/2} \right)
$$

#### **Intuition:**

- Mean difference $\to$ global shift
- Covariance difference $\to$ diversity & mode coverage
- Lower FID = closer generated distribution to real

#### **Why preferred:**

- Compares to real data
- Sensitive to both quality and diversity

Widely used for GANs, diffusion models, etc

#### **Limitations:**  

- Gaussian assumption in feature space.  
- Depends on feature extractor (typically Inception v3).

### 2.3 Learned Perceptual Image Patch Similarity (LPIPS)

While IS and FID focus on _distribution_ level, they don't directly measure **perceptual similarity** between individual generated images and real targets.

**LPIPS** (Zhang et al., 2018) - perceptual similarity metric, designed to better correlate with human visual judgments.

#### **Definition & Idea**

Given two images $ x $ and $ x' $, LPIPS compares **deep feature maps** extracted from a pretrained network (e.g., AlexNet, VGG, SqueezeNet):

1. Pass both images through a fixed pretrained network $ \phi $.  
2. Extract feature activations at multiple layers:  
   $$
   \phi^l(x), \; \phi^l(x') \quad \text{for layers } l=1,\dots,L
   $$
3. Normalize feature maps channel-wise: $ \hat{\phi}^l $
4. Compute:

$$
\mathrm{LPIPS}(x, x') = \sum_{l} w_l \cdot \| \hat{\phi}^l(x) - \hat{\phi}^l(x') \|_2^2
$$

#### **Intuition**

- Captures high-level perceptual structure rather than pixel-wise differences
- Lower LPIPS = more perceptually similar.

#### **Usage**

- Often reported alongside FID.

## 3. Autoencoders

Autoencoders - **generative models** that learn to **encode** inputs into a lower-dimensional **latent space** representation and **decode** back to reconstruct the original data.

- Encoder - layers of Residual blocks consisting of `nn.Conv*d`.
- Decoder - layers of Residual blocks consisting of `nn.Conv*d_transpose`.

### 3.1 Vanilla Autoencoder (AE)

**Architecture:**  

- Encoder: $ x \mapsto z = f_\theta(x) $  
- Decoder: $ z \mapsto \hat{x} = g_\phi(z) $  

Trained to **minimize reconstruction error**:

$$
\mathcal{L}_{\text{AE}}(\theta, \phi) = \mathbb{E}_{x \sim p_{\text{data}}} \big[ \| x - \hat{x} \|^2 \big]
$$

#### **Characteristics**

- (+) Fast  
- (-) Not probabilistic
- (-) Random samples often look **nonsensical**.
- (-) Generated samples tend to be **blurry** due to L2 reconstruction loss.

### 3.2 Variational Autoencoder (VAE)

**VAE**s (Kingma & Welling, 2014) introduce **probabilistic latent variables** and impose structure on the latent space.

- Encoder produces parameters of $ q_\phi(z|x) $ (typically Gaussian with mean and variance).  
- Decoder defines $ p_\theta(x|z) $.  
- Optimize **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}_{\text{VAE}}(\theta, \phi) =
\mathbb{E}_{x \sim p_{\text{data}}} \left[
    \mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)]
    - D_{\mathrm{KL}}(q_\phi(z|x) \,\|\, p(z))
\right]
$$

where $ p(z) $ is usually $ \mathcal{N}(0, I) $.

#### **Characteristics**

- (+) Fast.
- (+) Regularizes latent space via KL $\to$ **smooth latent manifold** $\to$ meaningful sampling.  
- (-) Still bluryy

### 3.3 Conditional VAE (CVAE)

You turn a VAE into a conditional model by feeding text $y$ into both encoder and decoder.

- **Text encoder**: tokenize → Transformer/BERT/CLIP-text → embedding $ e_y \in \mathbb{R}^d $

- **Encoder**: $ q_\phi(z \mid x, y) = \mathcal{N}(\mu_\phi(x, e_y), \mathrm{diag}(\sigma_\phi^2(x, e_y))) $

- **Prior**: either fixed $ p(z) = \mathcal{N}(0, I) $ or **conditional prior** $ p_\psi(z \mid y) $ (often a small MLP/flow taking $ e_y $)

- **Decoder**: $ p_\theta(x \mid z, y) $; inject $ e_y $ by concatenation, FiLM, or cross-attention

#### **Loss (per sample)**

$$
\mathcal{L} =
\underbrace{\mathbb{E}_{q_\phi(z\mid x,y)}[-\log p_\theta(x \mid z, y)]}_{\text{reconstruction}}
+ \beta
\underbrace{D_{\mathrm{KL}}\big(q_\phi(z \mid x,y) \,\|\, p_\psi(z \mid y)\big)}_{\text{regularization}}.
$$

### 3.4 Latent Diffusion with a VAE (AE used as latent space)

The standard text → image generation pipeline with autoencoders has **two distinct components**:

1. **Image Autoencoder (VAE or regular AE)**  
   - **Encoder** $E$: maps an image $x$ to a latent code $ z = E(x) $  
   - **Decoder** $D$: reconstructs the image from $z$: $ \hat{x} = D(z) $  
   - $z$ typically has **much lower spatial resolution and dimensionality** than the original image.

2. **Text-conditioned Generative Model** (e.g., diffusion, autoregressive Transformer)  
   - Operates **in the latent space**, not pixel space.  
   - Takes text embeddings (from CLIP, BERT, T5, etc.) and learns a distribution $ p(z \mid \text{text}) $.  
   - Generates new latent codes $\tilde{z}$ that correspond to images matching the text prompt.

Finally, the generated latent $\tilde{z}$ is passed through the decoder $D$ to produce the final image:

$$
\tilde{x} = D(\tilde{z}).
$$

## 4. Generative Adversarial Networks (GANs)

### 4.1 GAN

GANs (Goodfellow et al., 2014) frame generation as a **two-player game**:

- **Generator** $ G_\theta(z) $ maps noise $ z $ to data space.  
- **Discriminator** $ D_\phi(x) $ tries to distinguish real from fake.

$$
\min_\theta \max_\phi \; \mathbb{E}_{x \sim p_{\text{data}}} [\log D_\phi(x)]
+ \mathbb{E}_{z \sim p(z)} [\log (1 - D_\phi(G_\theta(z)))]
$$

#### **Characteristics**

- Fast
- High visual quality
- Low diversity: prone to **mode collapse**, where generator covers only a subset of modes in data distribution $\leftarrow$ direct implication of using TVD/KL/JS.

### 4.2 WGAN

Replacing JS with **Wasserstein distance** gives **non-vanishing gradients** even with disjoint supports:

$$
\min_\theta \max_{\phi \in \text{Lip-1}} \; \mathbb{E}_{x \sim p_{\text{data}}}[D_\phi(x)] - \mathbb{E}_{z \sim p(z)}[D_\phi(G_\theta(z))]
$$

Requires discriminator (called **critic**) to be **1-Lipschitz**.

#### **Techniques of Enforcing Lipschitz Constraint**

- **Weight clipping** (Arjovsky et al., 2017)  
  - Simply clip discriminator weights to a fixed box $[-c,c]$.  
  - Works but can lead to capacity underuse and optimization issues.

- **Gradient penalty** (Gulrajani et al., 2017)  
  - Add penalty on the **norm of discriminator gradients** w.r.t. interpolated samples:
  $$
  \mathcal{L}_{\text{GP}} = \lambda \, \mathbb{E}_{\hat{x} \sim P_{\hat{x}}} \big[ \big( \| \nabla_{\hat{x}} D(\hat{x}) \|_2 - 1 \big)^2 \big]
  $$

  - Encourages discriminator to be 1-Lipschitz without hard clipping.

- **Spectral normalization** (Miyato et al., 2018)  
  - Normalize each weight matrix by its **largest singular value** (spectral norm):

  $$
  W_{\text{SN}} = \frac{W}{\sigma_{\max}(W)}
  $$
