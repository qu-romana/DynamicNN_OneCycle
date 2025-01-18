Dynamic Pruning for Neural Network Lottery Tickets in a Single Training Cycle

This repository demonstrates a dynamic pruning method inspired by the Lottery Ticket Hypothesis (LTH), designed to identify sparse sub-networks (winning tickets) within a **single training cycle**. Unlike traditional methods requiring iterative pruning and retraining, this approach progressively increases sparsity during training, achieving competitive accuracy while minimizing computational overhead.

---

## Overview

### Motivation
Traditional LTH methods are computationally expensive, as they involve repeated pruning and retraining cycles. Inspired by biological synaptic pruning, this project introduces a dynamic pruning mechanism that eliminates the need for retraining by incorporating sparsity directly into the training loop. The method is simple, efficient, and well-suited for resource-constrained environments.

Dynamic Supermask: A Balanced Approach to Sparsity
The dynamic supermask starts with all weights fully active (mask set to 1), ensuring that every weight has the opportunity to contribute to the learning process in the early stages of training. This approach avoids the risk of prematurely discarding connections that might later prove critical for task performance. Early epochs play a pivotal role in establishing meaningful patterns in the network, and starting with a fully active mask allows the model to fully explore the parameter space.

As training progresses, the dynamic supermask iteratively refines the sparsity pattern by gradually masking weights based on their magnitudes. This dynamic adjustment ensures that the network adapts its structure in real-time to focus on the most important connections while reducing the computational overhead of redundant weights. By balancing an inclusive starting point with gradual task-specific adaptation, this method provides a seamless integration of sparsity into the training process, achieving both efficiency and robust performance under sparse conditions.


### Key Highlights
- Dynamic Sparsity: Gradual pruning of low-magnitude weights during training.
- High Accuracy with High Sparsity: Achieves up to 80% sparsity on CIFAR-10 and 77% on MNIST, maintaining competitive performance.
- Efficient Training: No iterative pruning or weight resetting required.

---

 Datasets

 CIFAR-10
- Description: RGB images (32×32 pixels) from 10 object categories.
- Results: Test accuracy of 91.98% to 94.91% at 80% sparsity.

 MNIST
- Description: Grayscale images (28×28 pixels resized to 32×32) of handwritten digits (0–9).
- Results: Test accuracy of 99.45% to 99.51% at 77% sparsity.

---

 Methodology

 Architectures
- ResNet-18: Used for most experiments due to its balance between computational efficiency and accuracy.
- ResNet-34: Used in one experiment for comparison with deeper architectures.

 Training Configurations
- Compared configurations with and without validation sets:
  - Without Validation Set: Utilizes the entire dataset for training, yielding higher test accuracy in some cases.
  - With Validation Set: Facilitates hyperparameter tuning but may slightly lower test accuracy due to reduced training data.

 Hyperparameter Choices
The following hyperparameters were systematically varied and analyzed:
- Batch Size: 32 and 128.
- Learning Rates: 0.001 and 0.1.
- Momentum: 0.5 and 0.9.
- Epochs: 30, 100, 200, and 400.

 Sparsity
- Incremental sparsity increase up to 80% (CIFAR-10) and 77% (MNIST) using a binary supermask.
- Weights were pruned based on their magnitudes, with smaller weights removed progressively during training.

---

 Results

 CIFAR-10
- Test Accuracy: 91.98% to 94.91%.
- Final Sparsity: 80%.

 MNIST
- Test Accuracy: 99.45% to 99.51%.
- Final Sparsity: 77%.

Both datasets achieved high accuracy while reducing the active parameters significantly, demonstrating the method's robustness and efficiency.

---

 Repository Structure

 Main Training Files
1. CIFAR10_With_Validation.ipynb: 
   - Training CIFAR-10 with a validation set to monitor accuracy and loss during training.

2. CIFAR10_Without_ValidationSet.ipynb:
   - Trained CIFAR-10 once without using a validation set, allowing for the use of all available data for training.

3. MNIST_with_Validation.ipynb:
   - Training MNIST with a validation set to facilitate hyperparameter tuning.

4. MNIST_without_ValidationSet.ipynb:
   - Trained MNIST once without a validation set, optimizing for the use of all data in training.

---

### Experiment Files (Hyperparameter Analysis)
1. Batch Size:
   - `Experiment_Cifar10_BatchSize_32.ipynb`: Evaluates the impact of a smaller batch size (32) on CIFAR-10 training stability and convergence.
   - `Experiment_MNIST_BS_32.ipynb`: Explores the effect of a smaller batch size (32) on MNIST.

2. Learning Rate:
   - `Experiment_Cifar10_LR_0.001.ipynb`: Training CIFAR-10 with a lower learning rate (0.001) to observe its effect on convergence.
   - `Experiment_Cifar10_LR_0.01.ipynb`: Training CIFAR-10 with a higher learning rate (0.1) for faster convergence.
   - `Experiment_MNIST_LR_0.001.ipynb`: Analyzes the impact of a lower learning rate (0.001) on MNIST.

3. Momentum:
   - `Experiment_MNIST_momentum0.5.ipynb`: Evaluates the effect of lower momentum (0.5) on MNIST convergence stability.
   - `Experiment_MNIST_EP_10_momentum0.5.ipynb`: Combines a reduced epoch count (10) with momentum (0.5) for analysis.

4. Epochs:
   - `Experiment_Cifar10_Epoch30_BatchSize_32.ipynb`: Training CIFAR-10 with 30 epochs and batch size 32 to observe underfitting tendencies.
   - `Experiment_Cifar10_Epoch400_BatchSize_128.ipynb`: Extends training to 400 epochs with batch size 128 for CIFAR-10.
   - `Experiment_MNIST_with_Epoch100.ipynb`: Evaluates MNIST performance with 100 epochs.

5. Architecture Comparison:
   - `Experiment_Cifar10_Resnet34.ipynb`: Compares ResNet-34 against ResNet-18 on CIFAR-10, focusing on deeper architectures' trade-offs.

---
Final Remarks
This project demonstrates the power of dynamic pruning in simplifying the sparsification process while maintaining high accuracy. The results highlight its potential to optimize neural networks for resource-constrained environments, making it a practical and efficient solution for real-world applications.
