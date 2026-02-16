Jacobian Descent
================

This guide briefly explains what Jacobian descent is and on what kind of problems it can be used.
For a more theoretical explanation, please read our article
[Jacobian Descent for Multi-Objective Optimization](https://arxiv.org/pdf/2406.16232).

**Introduction**

The goal of Jacobian descent is to train models with multiple conflicting losses. When you have
multiple losses, your options are:

- Gradient descent: Sum the losses into a single loss, compute the gradient of this loss with
  respect to the model parameters, and update them using this vector. This is the standard approach
  in the machine learning community.
- Jacobian descent: Compute the gradient of each loss (stacked in the so-called Jacobian matrix),
  **aggregate** that matrix into a single update vector, and update the model parameters using this
  vector.

There are many different ways to aggregate the Jacobian matrix. For instance, we may be tempted to
just sum its rows. By linearity of differentiation, this is actually equivalent to summing the
losses and then computing the gradient, so doing that is equivalent to doing gradient descent.

If you have two gradients with a negative inner product and quite different norms, their sum will
have a negative inner product with the smallest gradient. So, given a sufficiently small learning
rate, a step of gradient descent will **increase** that loss. There are, however, different ways of
aggregating the Jacobian leading to an update that has non-negative inner product with each
gradient. We call these aggregators **non-conflicting**. The one that we have developed ourselves
and that we recommend for most problems is :doc:`UPGrad <docs/aggregation/upgrad>`.

**Which problems are multi-objective?**

Many optimization problems are multi-objective. In multi-task learning, for example, the loss of
each task can be considered as a separate objective. More interestingly to us, many problems that
are traditionally considered as single-objective can actually be seen as multi-objective. Here are a
few examples:

- We can consider separately the loss of each element in the mini-batch, instead of averaging them.
  We call this paradigm instance-wise risk minimization (:doc:`IWRM <examples/iwrm>`).
- We can split a multi-class classification problem with M classes into M binary classification
  problems, each one with its own loss.
- When dealing with sequences (text, time series, etc.), we can consider the loss obtained for each
  sequence element separately rather than averaging them.

**When to use Jacobian descent?**

JD should be used to try new approaches to train neural networks, where GD generally struggles due
to gradient conflict. If you have an idea where JD could be interesting, you should start by
verifying that the pairwise inner products between your gradients are sometimes negative. To easily
do that, you can start by following the :doc:`Monitoring <examples/monitoring>` examples. In
general, the effect of JD will be even greater if the gradients also have norm imbalance. Then, you
should use TorchJD to solve this conflict, and look at training and testing metrics to see if this
helps to solve your problem.

**When not to use Jacobian descent?**

- If training efficiency is critical (e.g. you're training LLMs with billions of parameters), it's
  likely that the memory overhead of JD will not make it worthwhile.
- If you have too many (e.g. more than 64) losses, JD will likely take too much memory to store the
  full Jacobian, and the aggregation will be too long with most aggregators. In that case, you could
  try to average some of your losses so that you end up with a reasonable number of losses.
- If the inner products between pairs of gradients are never negative, you're most likely good to go
  with GD.

**Getting started**

To start using TorchJD, :doc:`install <installation>` it and read the :doc:`basic usage example
<examples/basic_usage>`.
