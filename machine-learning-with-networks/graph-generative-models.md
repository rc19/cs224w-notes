---
layout: post
title: Generative Models for Graphs
header-includes:
   - \usepackage{amsmath}
---

In the [Node Representation learning](node-representation-learning.md) section, we saw several methods to "encode" a graph in the embedding space while preserving the nodes' similarity and network structure. In this section, we will study how to express probabilistic dependencies among a graph’s nodes and edges and generate new realistic graphs by drawing samples from the leanred distribution. This ability to capture the distribution of a particular family of graphs has many applications. For instance, sampling from the graphmodel can lead to the discovery of new configurations that share same global properties as is, for example, required in drug discovery. Another application of these methods is the ability to simulate "What-If" scenarios in a real world graph to gather insights about the network properties and attributes.

## Challenges 
1. For a graph of n nodes, there are $$O(n^2)$$ possible edges which results in a quadratic explosion while predicting edges in the graph.
![quadratic_explosion](../assets/img/quadratic_explosion.png?style=centerme)
2. n-node graph can be represented in n! ways which makes it very hard to optimize objective functions as 2 very different adjacency matrix representations of graphs can result in the same graph structure.
![permutation_invariant](../assets/img/permutation_invariant.png?style=centerme)
3. Edge formation can have long-range dependencies (e.g. to generate a graph having a 6-node cycle, need to remember the structure so far)
![long_range_dependency](../assets/img/long_range_dependency.png?style=centerme)

**Terminology**
1. $$p_{data}(G)$$: Probability distribution from which a given graph is sampled.
2. $$p_{model}(G;\theta)$$: The distribution, parametrized by $$\theta$$, learned by the model to approximate $$p_{data}(G)$$

**Goal**: Our goal is 2-fold. 
1. Make sure that $$p_{model}(G;\theta)$$ is very close to $$p_{data}(G)$$ (Key Idea: Maximum Likelihood)
2. Furthermore, we also need to make sure that we can efficiently sample graphs from $$p_{model}(G;\theta)$$ (Key Idea: Sample from noise distribution and transfrom the sampled noise via a complex function to generate the graph)

## GraphRNN

