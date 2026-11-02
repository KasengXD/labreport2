import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# -------------------- Fixed Problem Definitions --------------------
# These values are fixed to satisfy the assignment requirements.
POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
TARGET_ONES = 50
GENERATIONS = 50

# GA operator settings retained from the supplied code.
CROSSOVER_RATE = 0.90
MUTATION_RATE = 0.01
TOURNAMENT_K = 3
ELITISM = 2


def fitness_function(chromosome: np.ndarray) -> float:
    """
    Fitness is highest when a chromosome contains exactly 50 ones.
    Required maximum: 80 - abs(50 - 50) = 80.
    """
    count_ones = int(np.sum(chromosome))
    return float(80 - abs(count_ones - TARGET_ONES))


# -------------------- GA Operators --------------------
def init_population(
    pop_size: int,
    chromosome_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a random population containing 300 binary chromosomes."""
    return rng.integers(
        0,
        2,
        size=(pop_size, chromosome_length),
        dtype=np.int8,
    )


def tournament_selection(
    fitness: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> int:
    """Select the fittest individual from k randomly selected candidates."""
    candidate_indices = rng.integers(0, fitness.size, size=k)
    best_candidate = candidate_indices[np.argmax(fitness[candidate_indices])]
    return int(best_candidate)


def one_point_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Exchange chromosome sections after one randomly selected point."""
    crossover_point = int(rng.integers(1, parent_a.size))

    child_a = np.concatenate(
        [parent_a[:crossover_point], parent_b[crossover_point:]]
    )
    child_b = np.concatenate(
        [parent_b[:crossover_point], parent_a[crossover_point:]]
    )
    return child_a, child_b


def bit_mutation(
    chromosome: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip each bit independently according to the mutation probability."""
    mutated = chromosome.copy()
    mutation_mask = rng.random(chromosome.shape) < mutation_rate
    mutated[mutation_mask] = 1 - mutated[mutation_mask]
    return mutated


def evaluate(population: np.ndarray) -> np.ndarray:
    """Calculate the fitness score for every chromosome in the population."""
    return np.array(
        [fitness_function(individual) for individual in population],
        dtype=float,
    )


# -------------------- Genetic Algorithm --------------------
def run_ga(seed: int) -> dict:
    """
    Execute exactly 50 generations with a population of 300 chromosomes.
    The best two individuals are retained through elitism.
    """
    rng = np.random.default_rng(seed)

    population = init_population(
        POPULATION_SIZE,
        CHROMOSOME_LENGTH,
        rng,
    )
    fitness = evaluate(population)

    best_fitness_history = []

    for _ in range(GENERATIONS):
        # Preserve the two fittest individuals before creating offspring.
        elite_indices = np.argsort(fitness)[-ELITISM:]
        elites = population[elite_indices].copy()

        next_population = []

        # Create enough offspring to keep the population size at 300.
        while len(next_population) < POPULATION_SIZE - ELITISM:
            # Tournament selection chooses the two parents.
            parent_1_index = tournament_selection(fitness, TOURNAMENT_K, rng)
            parent_2_index = tournament_selection(fitness, TOURNAMENT_K, rng)

            parent_1 = population[parent_1_index]
            parent_2 = population[parent_2_index]

            # Apply single-point crossover with the configured probability.
            if rng.random() < CROSSOVER_RATE:
                child_1, child_2 = one_point_crossover(
                    parent_1, parent_2, rng
                )
            else:
                child_1 = parent_1.copy()
                child_2 = parent_2.copy()

            # Apply bit-flip mutation to both offspring.
            child_1 = bit_mutation(child_1, MUTATION_RATE, rng)
            child_2 = bit_mutation(child_2, MUTATION_RATE, rng)

            next_population.append(child_1)
            if len(next_population) < POPULATION_SIZE - ELITISM:
                next_population.append(child_2)

        # Add the elite chromosomes to complete the next population.
        population = np.vstack([np.asarray(next_population), elites])
        fitness = evaluate(population)

        # Record the best score after each generation (50 values in total).
        best_fitness_history.append(float(np.max(fitness)))

    # Identify the best chromosome in the final population.
    best_index = int(np.argmax(fitness))
    best_chromosome = population[best_index].copy()
    best_fitness = float(fitness[best_index])

    return {
        "best": best_chromosome,
        "best_fitness": best_fitness,
        "history": pd.DataFrame(
            {
                "Generation": np.arange(1, GENERATIONS + 1),
                "Best Fitness": best_fitness_history,
            }
        ),
        "final_population": population,
        "final_fitness": fitness,
    }


# -------------------- Streamlit User Interface --------------------
st.set_page_config(
    page_title="Genetic Algorithm - Bit Pattern",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Genetic Algorithm: Bit Pattern Optimisation")
st.caption(
    "Evolve an 80-bit chromosome so that it contains exactly 50 ones."
)

# Display the assignment's fixed parameters in the sidebar.
with st.sidebar:
    st.header("Problem Parameters")
    st.markdown(
        f"""
        - **Population size:** {POPULATION_SIZE}
        - **Chromosome length:** {CHROMOSOME_LENGTH} bits
        - **Target number of ones:** {TARGET_ONES}
        - **Maximum fitness:** 80
        - **Generations:** {GENERATIONS}
        """
    )

    st.divider()
    st.subheader("GA Operator Settings")
    st.write(f"Crossover rate: {CROSSOVER_RATE:.2f}")
    st.write(f"Mutation rate: {MUTATION_RATE:.2f}")
    st.write(f"Tournament size: {TOURNAMENT_K}")
    st.write(f"Elitism: {ELITISM} individuals")

    st.divider()
    st.caption(
        "The assignment parameters are fixed and cannot be changed from "
        "the interface."
    )

# Seed input and algorithm execution.
seed = st.number_input(
    "Random seed (for reproducibility)",
    min_value=0,
    max_value=2**32 - 1,
    value=42,
    step=1,
)

if st.button("Run Algorithm", type="primary", use_container_width=True):
    with st.spinner("Running the Genetic Algorithm..."):
        result = run_ga(int(seed))

    # Save the result so it remains visible after Streamlit reruns.
    st.session_state["ga_result"] = result
    st.session_state["ga_seed"] = int(seed)

# Display results only after the user has run the algorithm.
if "ga_result" in st.session_state:
    result = st.session_state["ga_result"]
    best_chromosome = result["best"]

    count_ones = int(np.sum(best_chromosome))
    count_zeros = CHROMOSOME_LENGTH - count_ones
    bitstring = "".join(str(int(bit)) for bit in best_chromosome)

    st.subheader("Execution Summary")
    metric_1, metric_2, metric_3 = st.columns(3)

    metric_1.metric("Best Fitness Score", f"{result['best_fitness']:.0f} / 80")
    metric_2.metric("Count of 1s", f"{count_ones} / {CHROMOSOME_LENGTH}")
    metric_3.metric("Count of 0s", f"{count_zeros} / {CHROMOSOME_LENGTH}")

    st.subheader("Final Evolved Bitstring")
    st.code(bitstring, language="text")

    # Matplotlib convergence curve for the best fitness in each generation.
    st.subheader("Convergence Curve")
    history = result["history"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        history["Generation"],
        history["Best Fitness"],
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="Best fitness",
    )
    ax.axhline(
        y=80,
        linestyle="--",
        linewidth=1.2,
        label="Maximum fitness (80)",
    )
    ax.set_title("Best Fitness Across 50 Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Score")
    ax.set_xlim(1, GENERATIONS)
    ax.set_ylim(bottom=0, top=82)
    ax.set_xticks(range(0, GENERATIONS + 1, 5))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Optional table showing the final population and its fitness values.
    with st.expander("View final population (300 individuals)"):
        population_table = pd.DataFrame(
            result["final_population"],
            columns=[f"Bit {i + 1}" for i in range(CHROMOSOME_LENGTH)],
        )
        population_table.insert(
            0,
            "Individual",
            np.arange(1, POPULATION_SIZE + 1),
        )
        population_table["Fitness"] = result["final_fitness"]
        st.dataframe(population_table, use_container_width=True)

    if result["best_fitness"] == 80:
        st.success("The algorithm found a chromosome with exactly 50 ones.")
    else:
        st.info(
            "The algorithm completed all 50 generations. "
            "The best chromosome did not reach the target in this run; "
            "try another random seed."
        )
else:
    st.info("Enter a random seed and click **Run Algorithm** to start.")
