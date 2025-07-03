<!-- markdownlint-disable MD001, MD024, MD025, MD033, MD045 -->

# Lecture 1 - Intro

## Feedback Loop

**Feedback loop** - user interactions with recommendations (e.g. clicks, likes) influence future recommendations.

**Feedback loop problem** - model keeps reinforcing its own biases by learning only from user interactions on previous recommendations $\tp$ reduced diversity.

## 1. Collaboration-based Models

<div align="center"> <img src="notes_images/nu.png" width="400" height="70"> </div>

<div align="center"> <img src="notes_images/rui.png" width="250" height="70"> </div>

<div align="center"> <img src="notes_images/suv.png" width="400" height="200"> </div>

### User2User | Item2Item Recomendations

<div align="center"> <img src="notes_images/u2u_i2i.png" width="400" height="200"> </div>

## 2. Content-based Models

<div align="center"> <img src="notes_images/embed.png" width="400" height="250"> </div>

## 3. Hybrid Models

Collaboration + Content + Context (user-based, ..., weather, ...) + Business logic

## Ranking

<div align="center"> <img src="notes_images/ranking.png" width="400" height="250"> </div>

<div align="center"> <img src="notes_images/rankingmodel.png" width="400" height="250"> </div>

### Final Scheme

Base of Users & Content $\to$ Quick mechanism of candidate Users & Content selection $\to$ Ranking within candidates $\to$ Reranking $\to$ Recommendations

# Lecture 2 -

0:30:00