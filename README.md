# Stochastic Diffusion Search for Skin Lesion Localisation

## Overview

This project explores the use of **Stochastic Diffusion Search (SDS)**, a swarm intelligence algorithm, for skin-lesion localisation in image data.

The project was completed as part of the **Natural Computing** module and applies a natural computing technique to a real-world-inspired medical imaging problem.

SDS is used to demonstrate how a population of agents can explore an image search space, perform partial evaluations, and gradually cluster around potentially suspicious regions.

## Methodology

The SDS process is divided into two main phases:

1. **Test Phase**  
   Each agent evaluates a small local region of the image and becomes active if the region appears promising.

2. **Diffusion Phase**  
   Inactive agents either adopt the hypothesis of an active agent or generate a new random hypothesis.

Over repeated iterations, active agents concentrate around stronger candidate regions.


## Repository Structure

```text
.
├── code/
│   ├── SDS_main.py
│   ├── SDS sphere evaluation.py
│   ├── sds_sphere_per_agent_final.csv
│   ├── sds_sphere_summary.txt
│   └── skin_example.jpg
├── report/
│   └── sds-skin-cancer-report.pdf
├── LICENSE
└── README.md
```
## Academic Context

This mini project was completed for the **COMP1805 Natural Computing** module. The coursework focused on applying natural computing or swarm intelligence techniques to a selected problem and critically reflecting on their strengths, limitations, and real-world applicability.

## Notes

The weekly wiki reflections are included as supporting learning material. They document my understanding of topics such as the No Free Lunch theorem, Dispersive Flies Optimisation, Genetic Algorithms, and Stochastic Diffusion Search.

