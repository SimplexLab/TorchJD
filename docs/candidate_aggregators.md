# Candidate Aggregators for Multi-Task Learning

## GradNorm
GradNorm (Gradient Normalization) is a dynamic gradient-based approach that automatically balances task training by adjusting the weight coefficients of each task's loss. It aims to equalize the gradient norms of different tasks by minimizing an auxiliary loss. This auxiliary loss penalizes the difference between the actual task gradient norm and a target norm derived from the task's relative training rate. The main goal is to ensure that no single task dominates the model updates, thus preventing overfitting and enabling more effective learning across multiple tasks. It focuses primarily on dynamic gradient magnitude tuning.

## DB-MTL (Dual-Balancing Multi-Task Learning)
DB-MTL is a method that handles imbalances at both the loss level and the gradient level simultaneously through a "dual-balancing" strategy. For loss-scale balancing, it applies a parameter-free logarithm transformation on each task's loss to bring them to a similar scale. For gradient-magnitude balancing, it employs a training-free maximum-norm normalization strategy, which rescales all task gradients to have the same magnitude as the maximum gradient norm among the tasks. Unlike GradNorm, which uses an auxiliary loss and dynamic tuning, DB-MTL is computationally efficient (training-free) and effectively equalizes both loss and gradient scales.
