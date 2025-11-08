# CosmicBeats Simulator (Updated Installation Guide)

This repository is a fork of Microsoft's [CosmicBeats-Simulator](https://github.com/microsoft/CosmicBeats-Simulator). This README provides updated and validated installation instructions to ensure compatibility with modern development environments (like macOS or PCs with new Python versions).

The original simulation platform caters to diverse research interests, including networking, AI, computing, and more.

## ⚠️ Critical Prerequisites: Python 3.12

The project's dependencies have **critical incompatibilities** with the latest Python versions (e.g., Python 3.13+). Attempting to install with a newer Python version will result in compilation errors (specifically for `pydantic-core`, a sub-dependency).

To ensure a successful installation, you **must use Python 3.12**.

### Required Tools (macOS Example)
1.  **Homebrew** (for macOS): Used to install Python. [Install Homebrew here](https://brew.sh/).
2.  **Git:**
    ```bash
    brew install git
    ```
3.  **Python 3.12:**
    ```bash
    brew install python@3.12
    ```
---

## 🚀 Installation Guide

Please follow these steps exactly to set up your environment.

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone [https://github.com/wocn-unicamp/CosmicBeats-Simulator.git](https://github.com/wocn-unicamp/CosmicBeats-Simulator.git)
cd CosmicBeats-Simulator

### 2. Fix the requirements.txt File
The original requirements.txt file pins old package versions that conflict with modern Python. We must first overwrite it to allow pip to install the latest compatible versions.
Replace the entire content of your requirements.txt file with the following:

astropy
dask
geopy
gradio
numpy
pandas
plotly
scipy
statsmodels
tqdm