Assignment 1: Machine Learning Foundations & Responsible AI

Course Track: WiDS Kalman Filtered Trend Trader

This repository contains Assignment 1, which focuses on building strong mathematical, ethical, and implementation foundations for machine learning, with applications to quantitative finance and deep learning.

- Project Overview

The goal of this assignment is to move beyond black-box model usage and develop a first-principles understanding of:

 Linear regression and its mathematical derivation
 Algorithmic bias and fairness in predictive models
 Neural network architecture and training using PyTorch

The work emphasizes both theoretical rigor and practical implementation, along with responsible AI considerations.

- Linear Regression - Mathematical Foundations

 Geometric Interpretation
  Predicted values ŷ are the orthogonal projection of the target vector y onto the column space of the design matrix X.

 Bias-Variance Tradeoff
  Decomposed prediction error into:

   Bias²
   Variance
   Irreducible Error

 Multicollinearity
  Analyzed how correlated features lead to near-singular ( X^T X ), causing unstable coefficient estimates.

- Algorithmic Fairness & Bias Detection

 Fairness Metrics Implemented

   Demographic Parity Difference (DPD)
   Equal Opportunity Difference (EOD)
   Disparate Impact Ratio (DIR)

 Bias Analysis

   Demonstrated why removing protected attributes (e.g., gender) does not eliminate bias due to proxy variables
   Derived conditions for Omitted Variable Bias

 Model Interpretability

   Applied SHAP (SHapley Additive exPlanations) to understand feature influence and identify reliance on sensitive attributes

 - Deep Learning with PyTorch

 Neural Network Architecture

 Activation Functions

   Analyzed why ReLU outperforms sigmoid/tanh in deep networks by mitigating vanishing gradients

 Training Mechanics

   Used PyTorch's autograd system for automatic differentiation
   Understood the dynamic computation graph used in backpropagation

 Technologies Used

 - Math & Theory

   Linear Algebra
   Calculus
   Probability & Statistics

 - Libraries

   NumPy
   Pandas
   Scikit-learn
   SHAP
   PyTorch

- Outcomes

By completing this assignment, the project:

 Bridges mathematical theory and real-world ML implementation
 Demonstrates how bias can persist even in "fair-looking" models
 Builds intuition for neural network design and training dynamics
 Emphasizes responsible AI alongside performance

