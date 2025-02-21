---
title: 'Autoencoders for Research Paper Recommendation System'
date: 2025-1-20
permalink: /posts/2025/1/autoencoders/

---
Abstract
======

Recommendation systems have enjoyed widespread adoption in online commerce, news websites, and social media. Building recommendation systems is challeng- ing because high-dimensional data, such as article titles, need to be compressed into forms that computers can process. We propose a recommendation system for research papers that uses a paper’s title to recommend similar papers. To achieve this, we develop an unsupervised clustering algorithm on research paper titles using the autoencoder architecture. In particular, we use a recurrent neu- ral net (RNN) to “encode” paper titles as low-dimensional embeddings in latent space. We condition the autoencoder’s latent space geometry by reconstructing paper titles from their embeddings with a “decoder” RNN and back-propagating the reconstruction error signal through the network to minimize distances between structurally similar titles. Further, we aim to condition latent space geometry to minimize distances between conceptually similar titles by modifying the base- line model. The resulting model, called the Denoising Adversarial Autoencoder (DAAE), incorporates a denoising training objective by training with noisy train- ing data and an adversarial training objective by imposing a prior distribution on its latent variables. We compare this DAAE model with the baseline autoencoder model by performing k-nearest neighbor searches in their latent spaces, simulat- ing real-world usage of the recommendation system. Based on our findings, we determine that while the DAAE represents an improvement over the baseline au- toencoder, the recommendations are primarily helpful for junior researchers who have not established a focused research sub-field for their searches.


PDF Copy of Paper
======
Please see the PDF version of this research paper here: [paper](/files/autoencoders_research_paper_recommendation.pdf)
