# CogKG
This repository contains the source code and dataset for the paper: **Towards Unified Representations of Knowledge Graph and Expert Rules for Machine Learning and Reasoning**. 
Zhepei Wei, Yue Wang, Jinnan Li, Zhining Liu, Erxin Yu, Yuan Tian, Xin Wang and Yi Chang.
AACL 2022.

## Overview
With a knowledge graph and a set of if-then rules, can we reason about the conclusions given a set of observations?
In this work, we formalize this question as the \emph{cognitive inference} problem, and introduce the \underline{Cog}nitive \underline{K}nowledge \underline{G}raph (CogKG) that unifies two representations of heterogeneous symbolic knowledge: expert rules and relational facts.
We propose a general framework in which the unified knowledge representations can perform both learning and reasoning.
Specifically, we implement the above framework in two settings, depending on the availability of labeled data.
When no labeled data are available for training, the framework can directly utilize symbolic knowledge as the decision basis and perform reasoning.
When labeled data become available, the framework casts symbolic knowledge as a trainable neural architecture and optimizes the connection weights among neurons through gradient descent.
Empirical study on two clinical diagnosis benchmarks demonstrates the superiority of the proposed method over time-tested knowledge-driven and data-driven methods, showing the great potential of the proposed method in unifying heterogeneous symbolic knowledge, i.e., expert rules and relational facts, as the substrate of machine learning and reasoning models.

## Requirements
This repo was tested on Python 3.7 and the main requirements are:
- tqdm
- group-lasso==1.5.0
- interpret==0.2.7
- scikit-learn==1.0.2

## Datasets
-Muzhi
-MDD
