<!-- markdownlint-disable MD024 MD025 -->

# Notes

## Jupyter

On the server:

```bash
tmux ls
tmux attach -t cources
# OR
tmux new -s session

rm -rf ~/.jupyter
rm -rf ~/.local/share/jupyter
rm -rf ~/.ipython
rm -rf ~/.cache/jupyter
rm -rf ~/.config/jupyter

python3.11 -m venv venv311
source venv311/bin/activate

pip install jupyter

jupyter notebook --no-browser --port YOUR_PORT --ip 0.0.0.0
```

### Open in Browser

Locally:

```bash
ssh -N -f -L YOU_PORT:localhost:YOU_PORT shad-gpu
```

### Open in VSCode

1. Connect to server in VSCode.

2. Open .ipynb file and

<!-- markdownlint-disable MD024 MD025 -->

## Transferring Data

> Ctrl + Shift + P (Command Palette) $\to$ Remote-SSH: Kill VS Code Server on Host.

```bash
scp -r nlp_course/... shad-gpu:courses/nlp/nlp_course/...

scp -r shad-gpu:courses/nlp/nlp_course/... nlp_course/... 
```

# **Lecture 1 - Word Embeddings**

## Tokenization

- character (if you need to correct typos!)
- word
- group of words
- combined (sometimes char/word/group)

### Lookup Table

![alt text](notes_images/lookup_table.png)

For **Out-Of-Vocabulary** tokens $\to$ special `<UNK>` or split word by characters (since they are in vocab).

## Word Embedding: Count-Based Methods

### BoW (Bag of Words)

Not really a sentance embedding, rather high-dimensional sparse vector.

1. Build a vocabulary (**bag of words**) of all unique words.
2. Each sentance becomes a vector counting how many times each word appears.

Sentence: "cats eat fish"
Vocabulary: [cats, like, eat, fish, dogs]
Vector: [1, 0, 1, 1, 0]

### TF-IDF

Not all words are equally useful.

One can prioritize rare words and downscale words like "and"/"or" by using **tf-idf** (_text frequency/inverse document frequency_) features.

$$ \text{feature}_i = \frac{\text{Count}(word_i \in x)}{\text{Total number of words in } x} \times \log\left(\frac{N}{\text{Count}(word_i \in D) + \alpha}\right) $$

- $x$ is a single text
- $D$ is your dataset (a collection of texts)
- $N$ is a total number of documents
- $\alpha$ is a smoothing hyperparameter (typically 1).
- $Count(word_i \in D)$ is the number of documents where $word_i$ appears.

## Word Embedding: Prediction-Based Methods (Static)

**Static** word embeddings give the same vector no matter the sentence.

### Word2Vec (Mikolov et al., 2013)

> Learns embeddings by predicting nearby words (context window).

We want parameters $\theta$ that maximize likelihood:

$$
L(\theta) = \prod_{t=1}^{T} \prod_{\substack{-m \le j \le m \\ j \ne 0}} P(w_{t+j} \mid w_t, \theta)
$$

$$
J(\theta) = - \frac{1}{T} \sum_{t=1}^T \sum_{\substack{-m \le j \le m \\ j \ne 0}} \log P(w_{t+j} \mid w_t, \theta)
$$

#### Probability of Context Word

For central word $c$, context word $o$:

$$
P(o \mid c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}
$$

- $v_c$: embedding of central word.  
- $u_o$: embedding of context word.  

Log Simplification:

$$
J(\theta) = - \frac{1}{T} \sum_{t=1}^T \sum_{\substack{-m \le j \le m \\ j \ne 0}}
\Big( u_{w_{t+j}}^T v_{w_t} - \log \sum_{w \in V} \exp(u_w^T v_{w_t}) \Big)
$$

- **Positive term**: reward true context dot product.  
- **Negative term**: normalization penalty.  

#### Window size

- Larger $\to$ more topical similarities

- Smaller $\to$ more functional & syntatic similarities

#### Skip-Gram vs CBOW

1. Skip-Gram

    Given “cat”, predict {“the”, “sat”, “on”}.

    - Works well on small data, captures rare words better.

2. CBOW (Continuous Bag of Words)

    Given {“the”, “sat”, “on”}, predict “cat”.

### GloVe (Pennington et al., 2014)

> Learns embeddings by factorizing the global co-occurrence matrix of words.

#### Setup

We start from a word–context **co-occurrence matrix**:

- Vocabulary size: $ V $.  
- Let $ X_{ij} $ = number of times word $ j $ appears in the context of word $ i $.  
- Total co-occurrences: $ X_i = \sum_j X_{ij} $.  

From this we can define **probabilities**:  

$$
P_{ij} = \frac{X_{ij}}{X_i} \quad = \; \Pr(\text{context word } j \mid \text{word } i)
$$

#### Model formulation

We want a function of word vectors that relates to co-occurrence statistics.  
Let word $ i $ have vector $ w_i $, and context word $ j $ have vector $ \tilde{w}_j $.  

We aim for:

$$
F(w_i, \tilde{w}_j, b_i, \tilde{b}_j) \; \approx \; \log(X_{ij})
$$

where $ b_i, \tilde{b}_j $ are bias terms.  

Natural choice:

$$
w_i^\top \tilde{w}_j + b_i + \tilde{b}_j \; \approx \; \log(X_{ij})
$$

#### Loss function

$$
J = \sum_{i=1}^V \sum_{j=1}^V f(X_{ij}) \, \Big(w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log(X_{ij})\Big)^2
$$

- weighting function: $ f(x) = \begin{cases}
    \left(\frac{x}{x_{\max}}\right)^\alpha & \text{if } x < x_{\max} \\
    1 & \text{otherwise}
    \end{cases} $

  - prevents overweighting very frequent words (like “the”).  
  - ensures rare co-occurrences don’t dominate either.  

After training, the **final embedding for a word** is often taken as the sum or average of $ w_i $ and $ \tilde{w}_i $.

## Similarities Across Languages

1. Train embeddings for each language
2. Linearly map existing vocabulary translations using small dictionary
3. Extrapolate on other

![alt text](notes_images/similarities.png)

Our purpose is to learn such a linear transform $W$ that minimizes the Euclidean distance between $Wx_i$ and $y_i$ for some subset of word embeddings. Thus we can formulate the so-called **Procrustes problem**.

$$W^*= \arg\min_W ||WX - Y||_F$$

Which looks like we need to do simple LinReg!

![alt text](notes_images/similarities02.png)

### Orthogonal Procrustes problem

> Self-consistent linear mapping between semantic spaces should be orthogonal.

Linear transformations that preserve dot products and vector norms are exactly orthogonal transformations:

- Norm preservation: $\|Wx\| = \|x\|$
- Inner product preservation: $(Wx)^T (Wy) = x^T y$

$$W^* = \arg\min_W ||WX - Y||_F \text{, where: } W^TW = I$$

#### Expanding the Frobenius norm

$$
\|WX - Y\|_F^2 = \text{Tr}\big((WX - Y)^T (WX - Y)\big)
$$
$$
= \text{Tr}(X^T W^T W X) - 2\text{Tr}(X^T W^T Y) + \text{Tr}(Y^T Y)
$$
$$
= \text{Tr}(X^T X) - 2\text{Tr}(W^T YX^T) + \text{Tr}(Y^T Y)
$$

The first and last terms are constants (don’t depend on $W$).  

So minimizing the error is equivalent to **maximizing**:
$$
W^* = \arg\max_W \text{Tr}(W^T YX^T)
$$

#### SVD

---

For any real matrix $A \in \mathbb{R}^{m \times n}$, the **singular value decomposition (SVD)** is a factorization:

$$
A = U \Sigma V^T
$$

- $U, V$ are orthogonal matrices,  
- $\Sigma$ is diagonal with singular ($> 0$) values.

---

Let’s set:
$$
M = YX^T \in \mathbb{R}^{d \times d}
$$

Do an SVD:
$$
M = U \Sigma V^T
$$

Substitute SVD:
$$
\max_{W^T W = I} \text{Tr}(W^T U \Sigma V^T) = \text{Tr}(W^T U V^T \Sigma)
$$

#### Why $W^* = UV^T$

Since $\Sigma$ is diagonal with nonnegative entries, the maximum of $\text{Tr}(W^T U V^T \Sigma)$ is achieved when $W^T U V^T = I$.  

$$W^T U V^T = I$$
$$W^T = (U V^T)^{-1}$$
$$W^{-1} = (U V^T)^{-1}$$
$$W^* = U V^T$$

# **Lecture 2 - Language Modeling**

## Language Modelling

**Language model (LM)** estimates the probability of a sequence of words:

$$
P(w_1, w_2, \ldots, w_T)
$$

For practical purposes, **chain rule**:

$$
P(w_1, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})
$$

## N-gram Language Modelling

### Markov (n-gram) Approximation

$$
P(w_t \mid w_{1:t-1}) \approx P(w_t \mid w_{t-n+1:t-1})
$$

- **Bigram**: $P(w_t\mid w_{t-1})$
- **Trigram**: $P(w_t\mid w_{t-2},w_{t-1})$

### 2. MLE Estimation (and the sparsity problem)

Let $c(h, w)$ - **count** of word $w$ after context $h$ in the train.

For **history** $h = w_{t-n+1:t-1}$:

$$
P_{\text{MLE}}(w \mid h) = \frac{c(h,w)}{c(h)}
$$

- **Issue**: unseen $ (h,w) \to 0 $ probability $\to$ Need to **smooth**.

### 3. Laplace (Additive) Smoothing

$$
P_{\text{Lap}}(w \mid h) = \frac{c(h,w) + \alpha}{c(h) + \alpha\,|V|}
$$

- $\alpha=1$ - simple but over-smooths rare events.

### 4. Linear Interpolation

For trigram example:
$$
P_{\text{interp}}(w_t\mid w_{t-2},w_{t-1}) =
\lambda_3 \frac{c(w_{t-2},w_{t-1},w_t)}{c(w_{t-2},w_{t-1})}
+ \lambda_2 \frac{c(w_{t-1},w_t)}{c(w_{t-1})}
+ \lambda_1 \frac{c(w_t)}{\sum_{w'} c(w')}
$$
with $\lambda_i \ge 0,\ \sum \lambda_i = 1$.

- Tuning: EM on held-out, grid search, Bayesian optimization.

### 5. Stupid Back-off (SBO)

Define recursively for $n$-gram context $h$:
$$
P_{\text{SBO}}(w\mid h) =
\begin{cases}
\frac{c(h,w)}{c(h)}, & \text{if } c(h,w)>0 \\
\alpha \, P_{\text{SBO}}(w \mid h'), & \text{otherwise}
\end{cases}
$$

- $h'$: drop the earliest token in $h$; $\alpha\!\approx\!0.4$.
- Not a proper probability (can sum $>1$) but works well at scale.

### 6. Kneser–Ney (KN)

#### **6.1 Continuation probability (unigram base)**

Define **continuation count**:

$$
\text{ContCount}(w) \equiv \big|\{ w' \;:\; c(w',w) > 0 \}\big|
$$

The **continuation probability** is then

$$
P_{\text{cont}}(w) =
\frac{\text{ContCount}(w)}{\sum_{w'} \text{ContCount}(w')}
$$

**Intuition**:

- Words like “dog” occur in many different contexts $\to$ high continuation probability.  
- Words like _“Francisco”_ occur mostly after “San” $\to$ low continuation probability, even if frequent overall.

### 6.2 Discounted higher-order estimates

> Probabilities add-up to 1

$$
P_{\text{KN}}(w \mid h_n)
= \underbrace{\frac{\max\{ c(h_n,w) - D,\,0 \}}{c(h_n)}}_{\text{discounted MLE}}
\;+\;
\underbrace{\lambda(h_n)}_{\text{back-off weight}}
\underbrace{P_{\text{KN}}(w \mid h_{n-1})}_{\text{shorter history}}
$$

$$
P_{\text{KN}}(w \mid h_0) =
P_{\text{KN}}(w \mid \emptyset) =
P_{\text{cont}}(w) =
\frac{\text{ContCount}(w)}{\sum_{w'} \text{ContCount}(w')}
$$

where

- $ h_{n-1} $ is the **shortened context** [drop the earliest token]
- $ D $ - _discount_ constant [prevents overconfident estimates for rare n-grams]
- $ \lambda(h) $ - _normalization_ weight

To compute $ \lambda(h) $, we use the number of **distinct successors** of the context:

$$
\text{SuccCount}(h) \equiv \big|\{ w \;:\; c(h,w) > 0 \}\big|
$$

Then

$$
\lambda(h) = \frac{D \cdot \text{SuccCount}(h)}{c(h)}
$$

The recursion continues until we reach the unigram level:

## Basic NN-LM

### 1. N-grams $\to$ NN

Traditional n-gram LM: unseen n-grams have zero probability unless smoothed.

NN LM's goal remains $ P(w_{1:T}) = \prod_{t=1}^T P(w_t \mid w_{1:t-1}) $, but the conditional distribution is now **neural**.

### 2. Basic NN-LM

![alt text](notes_images/nnlm01.png)

Each word $ w $ is mapped to a vector $ e_w \in \mathbb{R}^d $.  
Given context $ h = (w_{t-n+1}, \dots, w_{t-1}) $ embeddings as input:

$$
x_t = [e_{w_{t-n+1}}; \, e_{w_{t-n+2}}; \, \dots; \, e_{w_{t-1}}]
$$

**Loss function** (negative log-likelihood):

$$
\mathcal{L} = - \sum_{t=1}^T \log P_\theta(w_t \mid w_{1:t-1})
$$

### 3. Weight Tying (Press & Wolf, 2017)

- **Input matrix** $E$: maps tokens to embeddings.
- **Output matrix** $W_{\text{out}}$: maps hidden states to logits.
- Both are size $|V| \times d$

Similar words should _ideally_ have similar input and output representations $\to$ **weight tying**.

$$
W_{\text{out}} = E^\top
$$

After weight tying:
$$
z_{t,w} = \langle h_t, e_w \rangle
$$

The logit is just the dot product between the **context representation** and the **word embedding**, like in energy-based models or word2vec.

![alt text](notes_images/weight_tying.png)

### 4. Output Token Sampling

Use **temperature** in softmax.

> Simple $\argmax$ sampling can lead to circular output:  
"the proposed method is based on the other hand , the proposed method is based on the other hand , the proposed method is based on the other hand , ..."

#### **4.1. Top-K**

Sample from top-K most probable output tokens.

#### **4.2. Top-p (Nucleus)**

Pick as least output tokens to cover p% of CDF.

#### **4.3. Beam Search**

Keep the top $k$ most probable partial _sequences_ (the “beam”) at each step.

> Nice use case: translation - keeps several translations of texts.

### 4. Evaluation

#### **4.1 Perplexity**

Standard metric for LMs:

$$
\text{Perplexity}_{\text{sentance}} = \exp\left( - \frac{1}{T} \sum_{t=1}^T \log P(w_t \mid h_t) \right) = P(w_1, \dots, w_T)^{-\frac{1}{T}} = \left( \prod_t P(w_t \mid h_t)\right)^{-\frac{1}{T}},
$$

$$
\text{Perplexity}_\text{corpus} = \exp\left(-\frac{1}{\sum_{s=1}^S T_s} \sum_{s=1}^S \sum_{t=1}^{T_s} \log p\bigl(w_t^{(s)} \mid h_t^{(s)}\bigr)\right)
$$

- Lower $\to$ better

- Interpreted as the model’s **effective branching factor**.
  - Perplexity 50 ≈ model chooses among 50 equally likely words on average

- Reflects average probability assigned to data

Best/Worst Log-Likelihood $\equiv$ Best/Worst Perplexiy

> You can't compare models with different number of tokens (like tokens=letters vs tokens=words)

# **Seminar 2**

## `nn.Embedding`

`nn.Embedding` - **trainable** lookup table that maps token IDs to dense vectors.

It is initialized with random weights (_Xavier_) and learns them during training.

Text is tokenized $\to$ converted to IDs $\to$ passed through nn.Embedding $\to$ turned into embeddings for the model.

```py
emb = nn.Embedding(vocab_size, embedding_dim)  
emb.weight # vocab_size X embedding_dim

token_ids = torch.tensor([2, 5, 7])
out = emb(token_ids)  # shape: (3, embedding_dim)
```

## `nn.Conv1d` with Text Embeddings

- Embedding shape: `batch_size`, `sequence_len`, `embedding_dim`

- `nn.Conv1d` expects: `batch_size`, `channels`, `sequence_len`

$\to$ do:

```py
x = emb(ids)
x = x.transpose(1, 2)
out = conv1d(x)
```

# **Lecture 3 - Seq2Seq, Attention**

## Conditional Language Modelling

$$
P(w_1, \ldots, w_T \mid x) = \prod_{t=1}^{T} P(w_t \mid w_{<t}, x)
$$

## 1. Vanilla Encoder–Decoder (Sutskever et al., 2014)

**Architecture:** RNN encoder $\to$ fixed vector $\to$ RNN decoder. No attention.

### Math

- Encoder reads inputs $ x_1, \dots, x_T $:

$$
h_t = \mathrm{RNN}_{\text{enc}}(h_{t-1}, x_t), \quad h_0 = 0
$$

- Final hidden state $ h_T $ is the **context vector** $ c $.

- Decoder generates outputs sequentially:

$$
s_t = \mathrm{RNN}_{\text{dec}}(s_{t-1}, y_{t-1}, c)
$$
$$
P(y_t \mid y_{<t}, x) = \mathrm{Softmax}(W_o s_t + b_o)
$$

### Intuition

- The entire input is compressed into a single fixed-length vector $ c $.
- Works reasonably for short sequences, but **fails for long sequences** due to the information bottleneck.

## 2. Bahdanau (Additive) Attention (Bahdanau et al., 2014)

**Architecture:** RNN encoder–decoder with additive attention.  
Attention is computed **before** the decoder RNN step (_pre-RNN attention_).

> Computed this way attention in encoder could be **bidirectional**

### Math

- Encoder produces hidden states $ h_1, \dots, h_T $.

- Alignment scores for each encoder step $ i $ at decoder step $ t $:

$$
e_{t,i} = v_a^\top \tanh(W_s s_{t-1} + W_h h_i)
$$

- Attention weights:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^T \exp(e_{t,j})}
$$

- Context vector:

$$
c_t = \sum_{i=1}^T \alpha_{t,i} h_i
$$

- Decoder uses $[c_t; y_{t-1}]$ to produce the next state and output.

### Intuition

- Replaces the fixed $ c $ with a **dynamic context vector** $ c_t $ at each time step.
- Additive scoring uses a small MLP $\to$ more flexible and stable on smaller datasets.
- Enables the model to **“attend” to relevant parts of the input** as it generates each output token.

![alt text](notes_images/bahdanau.png)

## 3. Luong (Multiplicative) Attention (Luong et al., 2015)

**Architecture:** RNN encoder–decoder with multiplicative attention.  
Attention is computed **after** the decoder RNN step (_post-RNN attention_).  
Uses more efficient dot-product or general scoring functions.

### Math

- Decoder computes provisional state:

$$
s_t = \mathrm{RNN}_{\text{dec}}(s_{t-1}, y_{t-1})
$$

- Alignment scores:
  - **Dot**: $ e_{t,i} = s_t^\top h_i $

  - **General**: $ e_{t,i} = s_t^\top W_a h_i $

- Attention weights:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^T \exp(e_{t,j})}
$$

- Context vector:

$$
c_t = \sum_{i=1}^T \alpha_{t,i} h_i
$$

- Final attentional hidden state:

$$
\tilde{s}_t = \tanh(W_c [c_t; s_t])
$$

### Intuition

- Computes the decoder state first, then aligns it with encoder states.
- More efficient and scalable, closer to the attention formulation later used in Transformers.

![alt text](notes_images/luong.png)

## 4. Transformer Attention (Vaswani et al., 2017)

**Architecture:** Encoder–decoder model without any RNN / LSTM / CNN / ..., do only attention.

So 2. and 3. did attention only _between_ encoder & decoder (that were RNN / ...), now encoder & decoder are attention themselves (and we also keep doing attention between).

> Check out notes from ml02.

## KV-Caching

![alt text](notes_images/kv_caching.png)

Transformer complexity:

- without caching: $O(\text{input len}^2 \cdot \text{embed size} + \text{output len}^3 \cdot \text{embed size})$

- with caching: $O(\text{input len}^2 \cdot \text{embed size} + \text{output len}^2 \cdot \text{embed size})$

# **Seminar 3**

## BPE (Byte Pair Encoding) Tokenization

1. Initialize vocabulary: all individual characters (or bytes).

2. Count frequencies of all adjacent symbol pairs.

3. Merge the most frequent pair into a new symbol.

4. Repeat steps 2–3 until reaching the desired vocabulary size.

## "SolidGoldMagikarp"

- Tokenizer includes it $\to$ it has its own token and embedding slot.

- Training data excludes it $\to$ its embedding never gets meaningfully updated (stays near-random initialization).

- Inference encounters it $\to$ the model uses this random embedding in attention layers $\to$ leads to unpredictable or bizarre generations.

## Loss

...

## Metric - BLEU

...

# **Lecture 4 - Transfer Learning**

![alt text](notes_images/transfer_learning_map.png)

## ELMo

ELMo $\to$ task-specific model

...

## BERT

Transformer $\to$ general use

...

### MLM (Masked Language Modelling)

LM (Language Modelling) sees only forward.

MLM sees all context except mask.

### NSP (Next Sentence Prediction)

RoBERTa didn' do NSP as BERT did.

...

# **Lecture 5 - LLMs, GPTs**

![alt text](notes_images/nlp_evolution.png)

![alt text](notes_images/gpt_evolution.png)

## In-Context Learning

...

## Chinchilla Rule

...

## Data Pipeline

1. Download all of the Internet. Common crawl: 250 billion pages, >1PB (>1e6 GB)
2. Text extraction from HTML (challenges: math, boilerplate)
3. Filter undesirable content (e.g. NSFW, harmful content, PII)
4. Deduplicates (url/document/line). E.g. all the headers/menus in forums are always same
5. Heuristic filtering. Rm low quality documents (e.g. # words, word length, outlier tokens)
6. Model based filtering. Predict if page could be referenced by Wikipedia.
7. Data mix. Classify data categories (code/books/entertainment) $\to$ Reweight domains using scaling laws to get high downstream performance.

## Emergent Abilities of LLM

Quantitative changes in a system $\to$ qualitative changes in behavior.

Starting from GPT-3:

- Contextual Understanding
- Zero-shot Learning
- Basic Reasoning
- Creative Composition
- Adaptability

## Scaling Law

...

## Relative Position Encoding

The problem with sin/cos or with learnable position embeddings is...

$$
\text{RelativeAttention} = ...
$$

### ALiBi Embeddings

...

![alt text](image.png)

### Rotary Embeddings

![alt text](image-1.png)

$m$ - token's index.

## FFN

...

```py
...
```

### Key-Value Interpretation

...

### Gated FFN

GLU, Bilinear, ReGLU, GEGLU, SwiGLU

...

# **Seminar 5**

## Masked Self-Attention

how many attentions are there??

# **Lecture 6 - In-Context Learning**

## Prompting

### Naive Prompting

After asked question LLM might answer with another question!

BLUM 7b

### Few-Shot Prompting

...

### Chain-Of-Thought Prompting

...

## Compositional generalization

...

## Instruction Fine-Tuning (SFT)

### Algorithm

...

### Scaling Instruction-Tuning

...

### Human-based VS LLM-based Generated Instructions

...
