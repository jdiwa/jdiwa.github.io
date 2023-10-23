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

 In this project from UC Berkeley's CS189 (Introduction to Machine Learning), I built a neural network from the ground up. I first a base-level understanding of the math (and biology) behind how neural networks operate and then building increasingly complex models.

 Neural networks, like brains, are made up of small constituent units called neurons. These neurons build connections that allow them to communicate with each other; this process . In the human brain, these connections are called synapses, while in the neural network they are represented by a constantly shifting array of weights where each value in the array corresponds to a synapse. In order to "learn", like all machine learning models, a neural network relies on optimizing a constraint called loss -- the less loss, the better your model performs. In order to reduce loss, we do a "forward pass" where training data is passed into the neural network. The network makes its predictions take the gradient (essentially a derivative) of the loss function and adjust our model accordingly, repeating this process over and over to train the neural network. However, this process might need to happen hundreds or thousands of times before our model becomes sufficiently predictive.

 In order to reduce this high computational workload, neural networks use a mathematical technique called backpropagation:

| ![Math](/assets/images/posts/neural-network/derivation.png) | 
|:--:| 
| *Similar to the chain rule from calculus, we can optimize this process by taking the gradient of each layer's input with respect to its output and "backpropagate" by starting at the last layer and working backwards, caching and reusing our results as we go.* |

To further optimize the process, we can take advantage of parallelism by "batching" training points (each of which is represented by a vector) and performing operations on mini-batches (each of which is represented by a matrix) instead of individual training points:

```python
def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Parameters
        ----------
        dLdY  derivative of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """

        # unpack the cache
        X = self.cache["X"]
        Z = self.cache["Z"]
        
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer

        dLdZ = self.activation.backward(Z, dLdY)

        dX = dLdZ @ self.parameters["W"].T
        dW = X.T @ dLdZ
        dB = np.sum(dLdZ, axis=0, keepdims=True)

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.
        self.gradients["W"] = dW
        self.gradients["b"] = dB

        return dX
```

