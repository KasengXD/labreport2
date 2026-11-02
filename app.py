
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# 1. FIXED PROBLEM PARAMETERS
# ============================================================
POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
TARGET_ONES = 50
GENERATIONS = 50

# GA operator settings
CROSSOVER_RATE = 0.90
MUTATION_RATE = 0.01
TOURNAMENT_K = 3
ELITISM = 2


# ============================================================
# 2. FITNESS FUNCTION
# ============================================================
def fitness_function(chromosome):
    count_ones = int(np.sum(chromosome))
    return float(80 - abs(count_ones - TARGET_ONES))


# ============================================================
# 3. GENETIC ALGORITHM OPERATORS
# ============================================================
def init_population(pop_size, chromosome_length, rng):
    """Generate a random population of binary chromosomes."""
    return rng.integers(
        0, 2,
        size=(pop_size, chromosome_length),
        dtype=np.int8
    )


def tournament_selection(fitness, k, rng):
    """Select the fittest individual from k random candidates."""
    candidate_indices = rng.integers(0, fitness.size, size=k)
    winner = candidate_indices[np.argmax(fitness[candidate_indices])]
    return int(winner)


def one_point_crossover(parent_a, parent_b, rng):
    """Exchange chromosome sections after one crossover point."""
    point = int(rng.integers(1, parent_a.size))

    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
    child_b = np.concatenate([parent_b[:point], parent_a[point:]])

    return child_a, child_b


def bit_mutation(chromosome, mutation_rate, rng):
    """Flip selected bits from 0 to 1 or 1 to 0."""
    mutated = chromosome.copy()
    mutation_mask = rng.random(chromosome.shape) < mutation_rate
    mutated[mutation_mask] = 1 - mutated[mutation_mask]

    return mutated


def evaluate(population):
    """Evaluate the fitness of every individual."""
    return np.array(
        [fitness_function(individual) for individual in population],
        dtype=float
    )


# ============================================================
# 4. GENETIC ALGORITHM
# ============================================================
def run_ga(seed, progress_bar=None, status_text=None):

    rng = np.random.default_rng(seed)

    # Initialise population
    population = init_population(
        POPULATION_SIZE,
        CHROMOSOME_LENGTH,
        rng
    )

    fitness = evaluate(population)
    history = []

    for generation in range(1, GENERATIONS + 1):

        # Elitism: preserve the two best individuals
        elite_indices = np.argsort(fitness)[-ELITISM:]
        elites = population[elite_indices].copy()

        next_population = []

        # Generate offspring
        while len(next_population) < POPULATION_SIZE - ELITISM:

            # Tournament selection
            p1_index = tournament_selection(fitness, TOURNAMENT_K, rng)
            p2_index = tournament_selection(fitness, TOURNAMENT_K, rng)

            parent_1 = population[p1_index]
            parent_2 = population[p2_index]

            # Single-point crossover
            if rng.random() < CROSSOVER_RATE:
                child_1, child_2 = one_point_crossover(
                    parent_1, parent_2, rng
                )
            else:
                child_1 = parent_1.copy()
                child_2 = parent_2.copy()

            # Bit-flip mutation
            child_1 = bit_mutation(child_1, MUTATION_RATE, rng)
            child_2 = bit_mutation(child_2, MUTATION_RATE, rng)

            next_population.append(child_1)

            if len(next_population) < POPULATION_SIZE - ELITISM:
                next_population.append(child_2)

        # Create next generation
        population = np.vstack([
            np.asarray(next_population),
            elites
        ])

        fitness = evaluate(population)

        # Record generation statistics
        best_index = int(np.argmax(fitness))
        best_chromosome = population[best_index]

        history.append({
            "Generation": generation,
            "Best Fitness": float(np.max(fitness)),
            "Average Fitness": float(np.mean(fitness)),
            "Worst Fitness": float(np.min(fitness)),
            "Best Count of Ones": int(np.sum(best_chromosome)),
            "Unique Chromosomes": int(
                np.unique(population, axis=0).shape[0]
            )
        })

        # Update progress bar
        if progress_bar is not None:
            progress_bar.progress(
                generation / GENERATIONS,
                text=f"Generation {generation}/{GENERATIONS}"
            )

        if status_text is not None:
            status_text.caption(
                f"Best fitness: {np.max(fitness):.0f}"
            )

    # Identify final best solution
    best_index = int(np.argmax(fitness))
    best_chromosome = population[best_index].copy()
    best_fitness = float(fitness[best_index])

    return {
        "best": best_chromosome,
        "best_fitness": best_fitness,
        "history": pd.DataFrame(history),
        "final_population": population.copy(),
        "final_fitness": fitness.copy(),
        "seed": seed
    }


# ============================================================
# 5. STREAMLIT PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Genetic Algorithm Lab",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Algorithm: Bit Pattern Optimisation")

st.caption(
    "An interactive Genetic Algorithm that evolves an 80-bit "
    "chromosome towards exactly 50 ones."
)


# ============================================================
# 6. SIDEBAR - PROBLEM PARAMETERS
# ============================================================
with st.sidebar:

    st.header("📋 Problem Parameters")

    st.metric("Population Size", POPULATION_SIZE)
    st.metric("Chromosome Length", "80 bits")
    st.metric("Target Number of Ones", TARGET_ONES)
    st.metric("Maximum Fitness", "80")
    st.metric("Generations", GENERATIONS)

    st.divider()

    st.subheader("⚙️ GA Operator Settings")

    st.write(f"**Crossover rate:** {CROSSOVER_RATE:.0%}")
    st.write(f"**Mutation rate:** {MUTATION_RATE:.0%}")
    st.write(f"**Tournament size:** {TOURNAMENT_K}")
    st.write(f"**Elitism:** {ELITISM} individuals")

    st.divider()

    st.caption(
        "The assignment parameters are fixed and cannot "
        "be modified through this application."
    )


# ============================================================
# 7. MAIN INPUT
# ============================================================
st.subheader("Run Genetic Algorithm")

seed = st.number_input(
    "Random Seed",
    min_value=0,
    max_value=2**32 - 1,
    value=42,
    step=1,
    help="Use the same seed to reproduce the same GA run."
)

run_button = st.button(
    "🚀 Run Algorithm",
    type="primary",
    use_container_width=True
)

if run_button:

    progress_bar = st.progress(0)
    status_text = st.empty()

    with st.spinner("Executing Genetic Algorithm..."):

        result = run_ga(
            seed=int(seed),
            progress_bar=progress_bar,
            status_text=status_text
        )

    st.session_state["ga_result"] = result

    progress_bar.progress(
        1.0,
        text="All 50 generations completed!"
    )

    status_text.success("Algorithm execution completed.")


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================
if "ga_result" in st.session_state:

    result = st.session_state["ga_result"]

    best_chromosome = result["best"]
    history = result["history"]

    count_ones = int(np.sum(best_chromosome))
    count_zeros = CHROMOSOME_LENGTH - count_ones

    bitstring = "".join(
        str(int(bit)) for bit in best_chromosome
    )

    st.divider()
    st.header("📊 Execution Summary")

    # Four execution metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Best Fitness Score",
        f"{result['best_fitness']:.0f} / 80"
    )

    col2.metric(
        "Count of 1s",
        f"{count_ones} / 80"
    )

    col3.metric(
        "Count of 0s",
        f"{count_zeros} / 80"
    )

    col4.metric(
        "Unique Individuals",
        f"{history.iloc[-1]['Unique Chromosomes']} / 300"
    )

    # Target achievement
    if result["best_fitness"] == 80:
        st.success(
            "Target achieved! The best chromosome contains exactly 50 ones."
        )
    else:
        st.info(
            "The algorithm completed 50 generations. "
            "The target has not been reached in this run."
        )

    # Final chromosome
    st.subheader("🧬 Final Evolved Bitstring")
    st.code(bitstring, language="text")

    st.caption(f"Random seed used: {result['seed']}")

    st.divider()


    # ========================================================
    # 9. FITNESS CONVERGENCE CURVE
    # ========================================================
    st.subheader("📈 Fitness Convergence Curve")

    st.caption(
        "Tracks the best, average and worst fitness scores "
        "through all 50 generations."
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(
        history["Generation"],
        history["Best Fitness"],
        marker="o",
        markersize=3,
        linewidth=2,
        label="Best Fitness"
    )

    ax.plot(
        history["Generation"],
        history["Average Fitness"],
        linestyle="--",
        linewidth=1.6,
        label="Average Fitness"
    )

    ax.plot(
        history["Generation"],
        history["Worst Fitness"],
        linestyle=":",
        linewidth=1.6,
        label="Worst Fitness"
    )

    ax.axhline(
        y=80,
        linestyle="-.",
        linewidth=1.2,
        label="Maximum Fitness (80)"
    )

    ax.set_title("Fitness Across 50 Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Score")
    ax.set_xlim(1, GENERATIONS)
    ax.set_ylim(50, 82)
    ax.set_xticks(range(0, GENERATIONS + 1, 5))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


    # ========================================================
    # 10. POPULATION DIVERSITY
    # ========================================================
    st.subheader("🧩 Population Diversity")

    st.caption(
        "Shows how many unique chromosomes remain "
        "as the population evolves."
    )

    diversity_fig, diversity_ax = plt.subplots(
        figsize=(11, 3.5)
    )

    diversity_ax.plot(
        history["Generation"],
        history["Unique Chromosomes"],
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Unique Chromosomes"
    )

    diversity_ax.set_title(
        "Population Diversity Across Generations"
    )

    diversity_ax.set_xlabel("Generation")
    diversity_ax.set_ylabel("Unique Chromosomes")
    diversity_ax.set_xlim(1, GENERATIONS)
    diversity_ax.set_ylim(0, POPULATION_SIZE + 10)
    diversity_ax.grid(True, alpha=0.3)
    diversity_ax.legend()

    diversity_fig.tight_layout()
    st.pyplot(diversity_fig)
    plt.close(diversity_fig)


    # ========================================================
    # 11. FINAL FITNESS DISTRIBUTION
    # ========================================================
    st.subheader("📊 Final Population Fitness Distribution")

    fitness_fig, fitness_ax = plt.subplots(
        figsize=(11, 3.5)
    )

    fitness_ax.hist(
        result["final_fitness"],
        bins=np.arange(29.5, 81.5, 1),
        edgecolor="black",
        rwidth=0.85
    )

    fitness_ax.set_title(
        "Fitness Distribution of Final Population"
    )

    fitness_ax.set_xlabel("Fitness Score")
    fitness_ax.set_ylabel("Number of Individuals")
    fitness_ax.set_xticks(range(30, 81, 5))
    fitness_ax.grid(axis="y", alpha=0.3)

    fitness_fig.tight_layout()
    st.pyplot(fitness_fig)
    plt.close(fitness_fig)


    # ========================================================
    # 12. DOWNLOAD RESULTS
    # ========================================================
    st.divider()
    st.subheader("📥 Download Results")

    download_col1, download_col2 = st.columns(2)

    # Download generation statistics
    history_csv = history.to_csv(
        index=False
    ).encode("utf-8")

    download_col1.download_button(
        "Download Generation Statistics (CSV)",
        data=history_csv,
        file_name="ga_generation_statistics.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Prepare final population
    population_df = pd.DataFrame(
        result["final_population"],
        columns=[
            f"Bit_{i + 1}"
            for i in range(CHROMOSOME_LENGTH)
        ]
    )

    population_df.insert(
        0,
        "Individual",
        np.arange(1, POPULATION_SIZE + 1)
    )

    population_df["Number_of_Ones"] = np.sum(
        result["final_population"],
        axis=1
    )

    population_df["Fitness"] = result["final_fitness"]

    population_csv = population_df.to_csv(
        index=False
    ).encode("utf-8")

    download_col2.download_button(
        "Download Final Population (CSV)",
        data=population_csv,
        file_name="ga_final_population.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # 13. GENERATION STATISTICS TABLE
    # ========================================================
    with st.expander("View Generation Statistics"):

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # 14. FINAL POPULATION TABLE
    # ========================================================
    with st.expander("View Final Population (300 Individuals)"):

        st.dataframe(
            population_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 15. ACADEMIC EXPLANATION
# ============================================================
with st.expander("📚 How the Genetic Algorithm Works"):

    st.markdown("""
    **1. Initialization**

    A population of 300 individuals is randomly generated.
    Each individual contains 80 binary genes, where each gene
    is either 0 or 1.

    **2. Fitness Evaluation**

    Each chromosome is evaluated according to its number of ones.
    A fitness score of 80 is achieved when the chromosome contains
    exactly 50 ones.

    **3. Tournament Selection**

    Three individuals are randomly selected from the population.
    The individual with the highest fitness becomes a parent.

    **4. Single-point Crossover**

    Two parent chromosomes exchange their bit sections after
    one randomly selected crossover point to produce two offspring.

    **5. Bit-flip Mutation**

    Each bit has a 1% probability of mutation. A selected bit
    changes from 0 to 1 or from 1 to 0.

    **6. Elitism**

    The two fittest individuals are preserved and transferred
    directly to the next generation.

    **7. Termination**

    The algorithm stops after completing exactly 50 generations.
    The chromosome with the highest fitness in the final
    population is presented as the best solution.
    """)


# ============================================================
# 16. FOOTER
# ============================================================
st.divider()

st.caption(
    "Artificial Intelligence Lab | Genetic Algorithm | "
    "Population: 300 | Chromosome: 80 bits | Generations: 50"
)
