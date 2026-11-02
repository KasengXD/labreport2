# Genetic Algorithm: Bit Pattern Optimisation

## Project Overview

This project implements a Genetic Algorithm (GA) using Python and Streamlit. The main objective is to evolve a population of binary chromosomes to obtain an 80-bit chromosome containing exactly 50 ones.

The application provides an interactive interface to execute the algorithm, display the best fitness score, show the final evolved chromosome, and visualise the convergence process across 50 generations.

## Problem Requirements

The Genetic Algorithm follows these fixed parameters:

| Parameter | Value |
|---|---:|
| Population Size | 300 |
| Chromosome Length | 80 bits |
| Target Number of Ones | 50 |
| Maximum Fitness | 80 |
| Number of Generations | 50 |

### Genetic Algorithm Settings

| Parameter | Value |
|---|---:|
| Crossover Rate | 0.90 |
| Mutation Rate | 0.01 |
| Tournament Size | 3 |
| Elitism | 2 |

## Fitness Function

The fitness function evaluates each chromosome based on the difference between its number of ones and the target value of 50.

```python
def fitness_function(chromosome):
    count_ones = int(np.sum(chromosome))
    return float(80 - abs(count_ones - 50))
```

The maximum fitness score is 80, which is achieved when the chromosome contains exactly 50 ones.

For example:

| Number of Ones | Fitness Score |
|---:|---:|
| 40 | 70 |
| 45 | 75 |
| 50 | 80 |
| 55 | 75 |
| 60 | 70 |

## Genetic Algorithm Operators

The application uses the following GA operators:

1. **Initialization** – Generates an initial population of 300 random binary chromosomes, each containing 80 bits.
2. **Tournament Selection** – Randomly selects three individuals and chooses the individual with the highest fitness score as a parent.
3. **Single-point Crossover** – Selects a random crossover point and exchanges chromosome sections between two parents to produce offspring.
4. **Bit-flip Mutation** – Randomly flips individual bits from 0 to 1 or from 1 to 0 based on the mutation rate.
5. **Elitism** – Preserves the two individuals with the highest fitness scores for the next generation.

The process continues until the algorithm completes all 50 generations.

## Application Features

- **Problem Parameters:** Displays the fixed population size, chromosome length, target number of ones, maximum fitness and number of generations.
- **Random Seed:** Allows users to enter a random seed for reproducibility.
- **Run Algorithm:** Executes the Genetic Algorithm when the button is clicked.
- **Execution Summary:** Displays the best fitness score, number of ones and number of zeros.
- **Final Evolved Bitstring:** Displays the best 80-bit chromosome obtained from the final population.
- **Convergence Curve:** Uses Matplotlib to visualise the best fitness score across all 50 generations.
- **Final Population:** Provides an expandable table showing all 300 individuals and their fitness scores.

## Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib

## Project Structure

```text
Genetic-Algorithm-Lab/
│
├── app.py
├── requirements.txt
└── README.md
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

Open the local URL displayed in the terminal:

```text
http://localhost:8501
```

Enter a random seed and click **Run Algorithm** to start the Genetic Algorithm.

## Deployment

The application can be deployed using Streamlit Community Cloud.

1. Upload `app.py`, `requirements.txt` and `README.md` to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using your GitHub account.
4. Select **Create app**.
5. Choose the GitHub repository and branch.
6. Set the main file path to `app.py`.
7. Click **Deploy**.

Streamlit Community Cloud will install the required dependencies from `requirements.txt` and host the application.

## Expected Outcome

The objective is to obtain a chromosome with exactly 50 ones and 30 zeros, achieving the maximum fitness score of 80.

The convergence curve illustrates how the best fitness score changes throughout the 50 generations.

**Note:** Achieving a fitness score of 80 is the optimisation target, but it is not guaranteed for every random seed.
