# CP-RPCA: Conformal Prediction for Robust PCA

This repository provides the implementation of **Conformal Prediction Robust PCA (CP-RPCA)**, a novel method that integrates the conformal prediction framework with Robust Principal Component Analysis (RPCA). Unlike traditional RPCA methods that rely on the "low-rank + sparse" model, CP-RPCA does not depend on the accuracy of this assumption. It provides per-element confidence intervals for the low-rank component and exhibits strong robustness to model misspecification.

## Features
- Implementation of CP-RPCA with two underlying RPCA algorithms:
  - **Fast-RPCA**: Based on the gradient descent approach from the paper ["Fast Algorithms for Robust PCA via Gradient Descent"](https://arxiv.org/abs/1605.07784).
  - **RPCA-ALM**: Implements Algorithm 1 from the paper ["Robust Principal Component Analysis?"](https://arxiv.org/abs/0912.3599).
- Numerical simulation experiments to evaluate the performance of CP-RPCA.
- Example application in **video background modeling** using the [CDnet 2014 dataset](https://ieeexplore.ieee.org/document/6910011).

## License
This project is licensed under the [MIT License](LICENSE).