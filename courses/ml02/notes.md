<!-- markdownlint-disable MD024 MD025 -->

# **Lecture 1 - Tips & Tricks**

## Working with Pre-trained Model

### Finetuning

### Transfer Learning

### Distillation

### Quantizattion

### Triplet loss

## Convergence

### Learning Rate Scheduler

Adam's adaptiveness is not enough.

`torch.optim.lr_scheduler.ReduceLROnPlateau` is nice!

> 2:18:00

### Warmup

Statistics gathering optimizers need a bit of first steps in order to see path way clearly.

### Loss Scaling

![alt text](notes_images/loss_scaling.png)

## Overfitting

### Label Smoothing

> Not that often used.

### Temperature

Use this like with Learning Rate Scheduler (inversed)!

### Noise

Adding noise to input data (especially good with images).

### Augmentations

> Don't do augmentations in validation/test sets.

# **Lecture 2 - ...**

Do BatchNorm _before_ DropOut.

> 1:00:00