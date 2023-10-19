---
title: "Neural Network"
date: 2023-10-18T00:00:00-04:00
categories:
  - Project
tags:
  - Machine Learning
  - Python
  - PyTorch
header:
  teaser: /assets/images/posts/neural-network/neural-net-structure.png

---

 In this project from UC Berkeley's CS189 (Introduction to Machine Learning), I built a neural network from the ground up. By first doing the math and then building increasingly complex models 

 Before I even started coding, I had to really understand the math (and biology) fundamental to how neural networks operate. Neural networks, like brains, consist of small units called neurons. These neurons form connections that allow them to "communicate"  Like all machine learning models, a neural network relies on optimizing a constraint called loss -- the less loss, the better your model performs. In order to reduce loss, I take the gradient (essentially a derivative) of the loss function and adjust our model accordingly, repeating this process over and over to train the neural network. HoIver, this process might need to happen hundreds or thousands of times before our model becomes sufficiently predictive.

 In order to reduce this high computational workload, neural networks use a mathematical technique called backpropagation:

| ![Math](/assets/images/posts/neural-network/derivation.png) | 
|:--:| 
| *Similar to the chain rule from calculus, we can optimize this process by taking the gradient of each layer's input with respect to its output and "backpropagate" by starting at the last layer and working backwards, caching and reusing our results as we go.* |