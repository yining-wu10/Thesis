# Master’s Thesis Project

This repository contains the code for my master’s thesis, **“Learning Patient Representations for Offline Reinforcement Learning in Healthcare,”** completed at the **TUM School of Management, Technical University of Munich**, for the degree of **Master of Science in Management and Digital Technology**.

The experiments investigate whether **Minimal Representation Learning (MRL)** can construct compact and interpretable patient-state representations from offline transition data while preserving the quality of treatment policies. The method is evaluated in a simulated sepsis-treatment environment and compared with Fitted Q Iteration as an offline reinforcement learning baseline.

## Main Experiment

The main experiment is implemented in:

* [Thesis_Final_Experiment.ipynb](https://github.com/yining-wu10/Thesis/blob/main/Thesis_Final_Experiment.ipynb)

It uses the following modules:

* [mrl](https://github.com/yining-wu10/Thesis/tree/main/mrl): implementation of the Minimal Representation Learning framework
* [sepsisSimDiabetes](https://github.com/yining-wu10/Thesis/tree/main/sepsisSimDiabetes): stochastic sepsis-treatment simulation environment

## Environment

The required Python packages are listed in [`requirements.txt`](https://github.com/yining-wu10/Thesis/blob/main/requirements.txt).

They can be installed with:

```bash
pip install -r requirements.txt
```

