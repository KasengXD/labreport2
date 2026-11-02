import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Enhanced GA Bit-Pattern Solver | BSD3513",
    page_icon="🧬",
    layout="wide"
)

# ==============================================================================
# SIDEBAR CONTROL PANEL
# ==============================================================================
st.sidebar.title("🧬 GA Control Panel")
st.sidebar.caption("BSD3513 Artificial Intelligence — Lab Report 2")

# Mode Selection
mode = st.sidebar.radio(
    "Configuration Profile",
    options=["📌 Lab Report Defaults (Strict)", "🧪 Custom Experimentation Mode"],
    help="Switch to Custom Mode to tune operators, rates, and parameters beyond default lab criteria."
)

if mode == "📌 Lab Report Defaults (Strict)":
    pop_size = 300          # Requirement 1
    target_ones = 50        # Requirement 2[cite: 2]
    chrom_len = 80          # Requirement 3[cite: 2]
    max_fitness = 80.0      # Requirement 4[cite: 2]
    n_generations = 50      # Requirement 5[cite: 2]
    selection_method = "Tournament"
    crossover_type = "Single-point"
    tournament_k = 3
    crossover_rate = 0.9
    mutation_rate = 1.0 / chrom_len
    elitism_count = 2
else:
    st.sidebar.subheader("Parameters")
    pop_size = st.sidebar.number_input("Population Size", min_value=50, max_value=1000, value=300, step=10)
    chrom_len = st.sidebar.number_input("Chromosome Length", min_value=10, max_value=200, value=80, step=5)
    target_ones = st.sidebar.number_input("Target Number of 1s", min_value=1, max_value=chrom_len, value=50, step=1)
    max_fitness = float(st.sidebar.number_input("Max Fitness Value", min_value=1.0, value=80.0, step=5.0))
    n_generations = st.sidebar.number_input("Generations", min_value=10, max_value=500, value=50, step=5)
    
    st.sidebar.subheader("Operators & Hyperparameters")
    selection_method = st.sidebar.selectbox("Selection Method", ["Tournament", "Roulette Wheel"])
    crossover_type = st.sidebar.selectbox("Crossover Method", ["Single-point", "Two-point", "Uniform"])
    tournament_k = st.sidebar.slider("Tournament Size (k)", 2, 10, 3)
    crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.9, 0.05)
    mutation_rate = st.sidebar.slider("Mutation Rate (per bit)", 0.0, 0.1, float(1.0 / chrom_len), 0.005)
    elitism_count = st.sidebar.slider("Elite Individuals Preserved", 0, 10, 2)

seed = st.sidebar.number_input("Random Seed (Reproducibility)", min_value=0, value=42, step=1)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Lab Verification:** Fixed defaults satisfy all 5 constraints in Question 2[cite: 2].")

# ==============================================================================
# GA CORE FUNCTIONS
# ==============================================================================
def fitness(individual: np.ndarray) -> float:
    """
    Evaluates chromosome fitness.
    Max fitness = max_fitness (80.0) when sum(individual) == target_ones (50).
    """
    ones = int(individual.sum())
    return max_fitness - abs(ones - target_ones)

def init_population(pop_size: int, chrom_len: int) -> np.ndarray:
    return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)

def tournament_selection(pop: np.ndarray, fits: np.ndarray, k: int) -> np.ndarray:
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()

def roulette_selection(pop: np.ndarray, fits: np.ndarray) -> np.ndarray:
    min_f = np.min(fits)
    shifted_fits = fits - min_f + 1e-6
    probs = shifted_fits / shifted_fits.sum()
    selected_idx = np.random.choice(len(pop), p=probs)
    return pop[selected_idx].copy()

def crossover(p1: np.ndarray, p2: np.ndarray, method: str, rate: float):
    if np.random.rand() > rate:
        return p1.copy(), p2.copy()
    
    length = len(p1)
    if method == "Single-point":
        pt = np.random.randint(1, length)
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])
    elif method == "Two-point":
        pt1, pt2 = sorted(np.random.choice(range(1, length), size=2, replace=False))
        c1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])
        c2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]])
    else:  # Uniform
        mask = np.random.rand(length) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
    return c1, c2

def mutate(individual: np.ndarray, rate: float) -> np.ndarray:
    mask = np.random.rand(len(individual)) < rate
    individual[mask] = 1 - individual[mask]
    return individual

def calculate_diversity(pop: np.ndarray) -> float:
    """Calculates average pairwise normalized Hamming distance across population."""
    sample_size = min(50, len(pop))
    sample = pop[np.random.choice(len(pop), sample_size, replace=False)]
    diffs = 0
    pairs = 0
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            diffs += np.sum(sample[i] != sample[j])
            pairs += 1
    return (diffs / (pairs * pop.shape[1])) if pairs > 0 else 0.0

# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================
st.title("🧬 Genetic Algorithm: Bit Pattern Optimization")
st.markdown("### Interactive Search Engine for Target Bit Count Problem")

# Requirement Status Display
st.markdown("#### 📋 Question 2 Requirement Checklist")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("1. Pop Size", f"{pop_size}", "Fixed: 300" if pop_size == 300 else "Custom")
c2.metric("2. Target 1s", f"{target_ones}", "Fixed: 50" if target_ones == 50 else "Custom")
c3.metric("3. Bit Length", f"{chrom_len}", "Fixed: 80" if chrom_len == 80 else "Custom")
c4.metric("4. Max Value", f"{max_fitness:.0f}", "Fixed: 80" if max_fitness == 80 else "Custom")
c5.metric("5. Generations", f"{n_generations}", "Fixed: 50" if n_generations == 50 else "Custom")

st.markdown("---")

run_btn = st.button("🚀 Run Genetic Algorithm Simulation", type="primary", use_container_width=True)

if run_btn:
    random.seed(seed)
    np.random.seed(seed)

    # Initialize Containers for Real-time Streaming
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    chart_col, stat_col = st.columns([2, 1])
    with chart_col:
        st.subheader("📈 Live Convergence Tracker")
        chart_placeholder = st.empty()
    with stat_col:
        st.subheader("⚡ Real-Time Statistics")
        live_metric_best = st.empty()
        live_metric_avg = st.empty()
        live_metric_div = st.empty()

    # Data logs
    pop = init_population(pop_size, chrom_len)
    history = []

    best_overall_ind = None
    best_overall_fit = -np.inf

    # Evolutionary Process Loop
    for gen in range(1, n_generations + 1):
        fits = np.array([fitness(ind) for ind in pop])

        best_idx = np.argmax(fits)
        gen_best_f = fits[best_idx]
        gen_avg_f = np.mean(fits)
        gen_worst_f = np.min(fits)
        gen_diversity = calculate_diversity(pop)

        if gen_best_f > best_overall_fit:
            best_overall_fit = gen_best_f
            best_overall_ind = pop[best_idx].copy()

        history.append({
            "Generation": gen,
            "Best Fitness": gen_best_f,
            "Average Fitness": gen_avg_f,
            "Worst Fitness": gen_worst_f,
            "Population Diversity": gen_diversity
        })

        # Real-time Stream UI Updates
        df_hist = pd.DataFrame(history)
        chart_placeholder.line_chart(
            df_hist.set_index("Generation")[["Best Fitness", "Average Fitness", "Worst Fitness"]]
        )
        
        live_metric_best.metric("Current Best Fitness", f"{gen_best_f:.2f} / {max_fitness:.0f}")
        live_metric_avg.metric("Average Population Fitness", f"{gen_avg_f:.2f}")
        live_metric_div.metric("Population Diversity", f"{gen_diversity:.4f}")
        
        status_text.text(f"Evolving Generation {gen} of {n_generations}...")
        progress_bar.progress(gen / n_generations)

        # Elitism & New Generation Creation
        E = max(0, min(elitism_count, pop_size))
        elites = pop[np.argpartition(fits, -E)[-E:]].copy() if E > 0 else np.empty((0, chrom_len), dtype=np.int8)

        new_pop = []
        while len(new_pop) < (pop_size - E):
            # Selection
            if selection_method == "Tournament":
                p1 = tournament_selection(pop, fits, tournament_k)
                p2 = tournament_selection(pop, fits, tournament_k)
            else:
                p1 = roulette_selection(pop, fits)
                p2 = roulette_selection(pop, fits)

            # Crossover & Mutation
            c1, c2 = crossover(p1, p2, crossover_type, crossover_rate)
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            
            new_pop.append(c1)
            if len(new_pop) < (pop_size - E):
                new_pop.append(c2)

        pop = np.vstack([np.array(new_pop, dtype=np.int8), elites]) if E > 0 else np.array(new_pop, dtype=np.int8)

    status_text.success(f"✅ Evolution Complete across {n_generations} Generations!")

    # ==============================================================================
    # RESULTS AND ANALYSIS VISUALIZATIONS
    # ==============================================================================
    st.markdown("---")
    st.subheader("🏁 Optimization Results & Best Chromosome")

    ones_count = int(best_overall_ind.sum())
    zeros_count = chrom_len - ones_count
    bitstring = "".join(map(str, best_overall_ind.tolist()))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Final Best Fitness", f"{best_overall_fit:.0f} / {max_fitness:.0f}")
    r2.metric("Ones Count (Target = 50)", f"{ones_count}")
    r3.metric("Zeros Count", f"{zeros_count}")
    r4.metric("Optimal Match Accuracy", f"{(1 - abs(ones_count - target_ones)/chrom_len)*100:.1f}%")

    st.markdown("**Best Evolved Bitstring Structure:**")
    st.code(bitstring, language="text")

    # Visual Inspection Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Matplotlib Detailed Plot", "🧬 Top Population Heatmap", "💾 Export Simulation Data"])

    with tab1:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        df_log = pd.DataFrame(history)
        
        ax1.plot(df_log["Generation"], df_log["Best Fitness"], label="Best Fitness", color="#1f77b4", linewidth=2.5)
        ax1.plot(df_log["Generation"], df_log["Average Fitness"], label="Average Fitness", color="#ff7f0e", linestyle="--")
        ax1.plot(df_log["Generation"], df_log["Worst Fitness"], label="Worst Fitness", color="#d62728", linestyle=":")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness Score")
        ax1.set_title("Fitness Trajectory across 50 Generations")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right")

        st.pyplot(fig)

    with tab2:
        st.caption("Visualizing bit distribution (1s = Yellow, 0s = Purple) for the top 15 individuals in the final population.")
        final_fits = np.array([fitness(ind) for ind in pop])
        top_indices = np.argsort(final_fits)[-15:][::-1]
        top_chromosomes = pop[top_indices]

        fig_heat, ax_heat = plt.subplots(figsize=(12, 4))
        cax = ax_heat.imshow(top_chromosomes, cmap="viridis", aspect="auto")
        ax_heat.set_xlabel("Gene Position (0 to 79)")
        ax_heat.set_ylabel("Top Individual Rank")
        ax_heat.set_title("Gene Heatmap of Top 15 Chromosomes")
        fig_heat.colorbar(cax, orientation="horizontal", pad=0.2, label="Gene Value (0 or 1)")
        st.pyplot(fig_heat)

    with tab3:
        csv_data = pd.DataFrame(history).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Evolution History Log (CSV)",
            data=csv_data,
            file_name=f"GA_Lab2_Seed{seed}_Results.csv",
            mime="text/csv",
            use_container_width=True
        )
