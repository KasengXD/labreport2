# Genetic Algorithm: Bit Pattern Optimisation

## Project Overview

This project implements a Genetic Algorithm (GA) using Python and Streamlit. The objective is to evolve a population of binary chromosomes to obtain an 80-bit chromosome containing exactly 50 ones, achieving a maximum fitness score of 80.

The application provides an interactive web dashboard to execute the algorithm, track execution metrics in real time, inspect evolved bitstrings, visualize convergence and population diversity across 50 generations, analyze final population distributions, and export results as CSV files.

## Problem Requirements

The Genetic Algorithm strictly adheres to the following fixed parameters:

| Parameter | Value |
| --- | --- |
| Population Size | 300 |
| Chromosome Length | 80 bits |
| Target Number of Ones | 50 |
| Maximum Fitness Score | 80 |
| Number of Generations | 50 |

### Genetic Algorithm Settings

| Parameter | Value |
| --- | --- |
| Crossover Rate | 0.90 (90%) |
| Mutation Rate | 0.01 (1%) |
| Tournament Size ($k$) | 3 |
| Elitism Count | 2 individuals |

## Fitness Function

The fitness function evaluates each 80-bit chromosome based on the absolute difference between its count of ones and the target value of 50:

```python
def fitness_function(chromosome):
    count_ones = int(np.sum(chromosome))
    return float(80 - abs(count_ones - TARGET_ONES))

```

The maximum fitness score is 80, which is achieved when the chromosome contains exactly 50 ones.

| Number of Ones | Absolute Difference $\vert{}N_{\text{ones}} - 50\vert{}$ | Fitness Score ($80 - \vert{}N_{\text{ones}} - 50\vert{}$) |
| --- | --- | --- |
| 30 | 20 | 60 |
| 40 | 10 | 70 |
| 45 | 5 | 75 |
| **50** | **0** | **80 (Target)** |
| 55 | 5 | 75 |
| 60 | 10 | 70 |

## Genetic Algorithm Operators

1. **Initialization** – Generates a random binary population of 300 individuals using `numpy.random.default_rng`, each individual formatted as an 80-bit array of integers ($0$ or $1$).
2. **Fitness Evaluation** – Sums the ones in each chromosome and applies the linear fitness formula.
3. **Elitism** – Identifies and copies the top 2 fittest individuals directly into the next generation without modification.
4. **Tournament Selection** – Samples 3 random candidates ($k=3$) from the population and selects the candidate with the highest fitness score as a parent.
5. **Single-point Crossover** – With a 90% probability, splits two parent chromosomes at a randomly selected bit index ($1$ to $79$) and exchanges their segments to create two offspring.
6. **Bit-flip Mutation** – With a 1% per-bit probability, flips selected bits ($0 \rightarrow 1$ or $1 \rightarrow 0$).
7. **Termination** – Halts automatically after completing 50 generations and selects the individual with the highest fitness score.

## Application Features

* **Sidebar Configuration Display:** Displays fixed problem parameters, operator rates, and a confirmation note regarding parameter constraints.
* **Reproducible Execution:** Features a seed input control ($0$ to $2^{32}-1$, default `42`) to reproduce exact execution paths.
* **Execution Summary Dashboard:**
* **Metric Cards:** Best Fitness Score (`/ 80`), Count of 1s (`/ 80`), Count of 0s (`/ 80`), and Unique Individuals (`/ 300`).
* **Target Status Banner:** Notifies whether the 50-ones target was reached or if a near-optimal solution was obtained.
* **Final Bitstring Viewer:** Formats the best 80-bit chromosome inside a monospaced code container.


* **Interactive Visualizations:**
* **Fitness Convergence Curve:** Matplotlib chart plotting Best Fitness, Average Fitness, Worst Fitness, and the Maximum Fitness threshold (80) across all 50 generations ($y \in [50, 82]$).
* **Population Diversity Tracking:** Line graph displaying the count of unique chromosomes per generation to monitor genetic convergence.
* **Final Population Distribution:** Histogram charting the fitness distribution of all 300 final individuals across bins from 30 to 80.


* **Data Export Utilities:**
* `Download Generation Statistics (CSV)`: Exports generation-by-generation tracking metrics (`Generation`, `Best Fitness`, `Average Fitness`, `Worst Fitness`, `Best Count of Ones`, `Unique Chromosomes`).
* `Download Final Population (CSV)`: Exports full chromosome data for all 300 individuals, including individual IDs, bit columns (`Bit_1` through `Bit_80`), ones count, and final fitness.


* **Expandable Data Viewers & Guides:**
* Interactive Pandas DataFrames for Generation Statistics and Final Population.
* Built-in academic walkthrough explaining the 7 main stages of the algorithm.



## Technologies Used

* **Python** – Core programming language
* **Streamlit** – Web application UI framework
* **NumPy** – Vectorized array operations and random generation
* **Pandas** – Data collection, structuring, and CSV generation
* **Matplotlib** – Data visualization and plotting

## Project Structure

```text
Genetic-Algorithm-Lab/
│
├── app.py                  # Main Streamlit application file
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation

```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Genetic-Algorithm-Lab.git

```

### 2. Navigate to the Project Folder

```bash
cd Genetic-Algorithm-Lab

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Streamlit Application

```bash
streamlit run app.py

```

### 5. Open the Application

Navigate to the local URL displayed in your terminal (typically `http://localhost:8501`). Enter a random seed and click **🚀 Run Algorithm**.

## Deployment

The application can be deployed directly to Streamlit Community Cloud:

1. Commit and push `app.py`, `requirements.txt`, and `README.md` to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **Create app**.
4. Select your repository, branch, and set the main file path to `app.py`.
5. Click **Deploy**.

## Expected Outcome

The optimization target is to evolve an 80-bit individual containing **50 ones** and **30 zeros**, attaining the maximum fitness score of **80**.

The convergence curve and diversity plot illustrate how the population transitions from a high-diversity initial state toward an optimized, converged state over 50 generations.
