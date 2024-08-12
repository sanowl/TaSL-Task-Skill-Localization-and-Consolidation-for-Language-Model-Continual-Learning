# TaSL: Task Skill Localization and Consolidation

## Overview

This repository implements the Task Skill Localization and Consolidation (TaSL) framework for language model continual learning. TaSL enhances knowledge transfer between tasks without memory replay, addressing catastrophic forgetting and enabling efficient knowledge transfer in continual learning scenarios.

## Framework Architecture

The TaSL framework is structured as a modular system, encompassing various interconnected components that work in concert to facilitate effective continual learning. These components are designed to handle skill representation, importance assessment, localization, and consolidation processes, forming a comprehensive solution for adaptive learning in language models.

## Key Features

- Flexible skill unit representation supporting matrix-level and LoRA-tailored approaches
- Advanced importance calculation methods utilizing gradient-based techniques
- Sophisticated skill localization and consolidation strategies
- Integration of LoRA orthogonality regularization
- Configurable architecture adaptable to diverse learning contexts

## Implementation

The framework is implemented in Python, leveraging PyTorch for efficient deep learning operations. It is designed to seamlessly integrate with existing machine learning pipelines and can be easily adapted to various model architectures and training regimes.

## Requirements

- PyTorch
- NumPy

## Usage

The TaSL framework can be initialized and incorporated into existing training pipelines for continual learning tasks. Detailed usage instructions and API documentation are provided in the accompanying documentation.

## Citation

This implementation is based on the following paper:

Feng, Y., Chu, X., Xu, Y., Lu, Z., Liu, B., Yu, P. S., & Wu, X. M. (2024). TaSL: Task Skill Localization and Consolidation for Language Model Continual Learning. arXiv preprint arXiv:2408.05200.

For a comprehensive understanding of the theoretical foundations and empirical results, please refer to the original publication.
