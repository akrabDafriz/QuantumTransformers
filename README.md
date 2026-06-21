# Quantum Transformers for Text Classification: An Analysis of VQC Designs

This project explores the application of Quantum Transformers for Natural Language Processing (NLP), specifically focusing on text classification. It is a fork of the original Quantum Vision Transformer research, heavily modified to analyze how different Variational Quantum Circuit (VQC) designs impact the training dynamics, convergence, and overall performance of the model.

While classical Transformers have dominated NLP, Quantum Machine Learning (QML) presents a potential paradigm shift. This repository isolates the `MultiHeadSelfAttention` and `FeedForward` components of the Transformer architecture, injecting parameterized quantum circuits (implemented via TensorCircuit and JAX) to evaluate multiple entanglement and rotational strategies.

## Experimental VQC Designs

A core contribution of this project is the comparison of five distinct quantum circuit architectures applied to the transformer layers. Each design tests a different hypothesis regarding rotational complexity and entanglement capabilities:

Classical Baseline: A standard, non-quantum Transformer layer for performance benchmarking.

Basic VQC: A standard quantum layer using fundamental RX rotations and CNOT entanglement.

Design 1 (RY + CNOT): Tests the effect of RY rotations combined with sequential CNOT entanglement.

Design 2 (Hadamard + RX + CNOT): Introduces a Hadamard basis change prior to RX rotations and CNOT entanglement to evaluate superposition benefits.

Design 3 (Full Rotations + CNOT): Employs full single-qubit rotations (RX, RY, RZ) coupled with CNOT gates, representing a higher rotational complexity.

Design 4 (RX + CRX): Utilizes RX rotations alongside trainable Controlled-RX (CRX) gates to test the efficacy of trainable, dynamic entanglement.

## Datasets

The models are evaluated against a mix of classic NLP sentiment analysis tasks and custom datasets:

1. Sentiment Analysis (Tokenized with BPE Tokenization): **IMDb** Reviews, **Amazon** Cells, and **Yelp**. Each of these datasets contains 500 positive and 500 negative reviews, made by [Kotzias et al.](https://dl.acm.org/doi/10.1145/2783258.2783380), and can be found [here](https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set).
3. Custom Classifications (Tokenized with Word Level Tokenization): Custom **MC** and **RP** datasets. These datasets are created specifically for meaning classification and relative pronoun classification tasks, respectively, in QNLP context. Therefore, the sizes are quite small. They were made by [Lorenz et al.](http://dx.doi.org/10.1613/jair.1.14329) and can be found [here](https://github.com/CQCL/qnlp_lorenz_etal_2021_resources)

## Results and Observations

The research results and publication are on their way. This readme shall later be updated once the publication is done.

## Project Structure

- `quantum_transformers/`: The library source code.
  - `datasets.py`: Handles tokenization (BPE/WordLevel) and PyTorch DataLoader generation.
  - `transformers.py`: Contains the Flax-based Transformer architecture integrating quantum layers.
  - `quantum_layer.py`: Houses the logic for translating inputs to quantum states and defines the 5 custom VQC designs.
- `notebooks/quantum/run_experiments.py`: The primary training pipeline loop for batch execution of experiments.

## Acknowledgements

I would like to express my gratitude to Universitas Indonesia for providing the hardware and computational resources necessary for the training and evaluation of these models.

Additionally, this project builds upon the foundational Quantum Vision Transformers repository. Special thanks to Marçal Comajoan Cara, who developed the original architecture and QML framework evaluations as part of Google Summer of Code (GSoC) 2023 with the ML4SCI organization. Their robust codebase paved the way for this text classification research.

## License

The project is licensed under the GNU General Public License v3.0.