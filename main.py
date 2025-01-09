import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

def read_job_operations_from_file(file_name):
    job_operations = []
    with open(file_name, 'r') as file:
        for line in file:
            values = list(map(int, line.strip().split()))
            operations = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
            job_operations.append(operations)
    return job_operations

def initialize_pop(job_operations, population_size):
    population = []
    all_job_tasks = [(j, t) for j in range(len(job_operations)) 
                     for t in range(len(job_operations[j]))]
    
    for _ in range(population_size):
        individual = all_job_tasks.copy()
        random.shuffle(individual)
        population.append(individual)
    return population

def decode_chromosome(chromosome, job_operations):
    machine_schedules = {}
    machine_times = {}
    job_end_times = {j: 0 for j in range(len(job_operations))}
    
    # Initialize machine schedules and times
    max_machine_id = max(machine for job in job_operations for machine, _ in job)
    for m in range(max_machine_id + 1):
        machine_schedules[m] = []
        machine_times[m] = 0
    
    for job_id, task_id in chromosome:
        machine, processing_time = job_operations[job_id][task_id]
        job_start_time = job_end_times[job_id]
        task_start_time = max(machine_times[machine], job_start_time)
        task_end_time = task_start_time + processing_time
        
        machine_schedules[machine].append({
            'job': job_id,
            'task': task_id,
            'start': task_start_time,
            'end': task_end_time
        })
        machine_times[machine] = task_end_time
        job_end_times[job_id] = task_end_time
    
    # Return three values: machine schedules, machine times, and total processing time
    total_processing_time = max(machine_times.values())
    return machine_schedules, machine_times, total_processing_time

def genetic_algorithm(job_operations, population_size=50, max_generations=5000, 
                     crossover_rate=0.8, mutation_rate=0.05, plateau_threshold=50, 
                     selection_method="tournament", crossover_method="two_point", 
                     mutation_method="swap", elitism=True):
    """
    Main genetic algorithm with configurable operators and elitism.
    """
    population = initialize_pop(job_operations, population_size)
    best_fitness = float('inf')
    best_solution = None
    fitness_history = []  # List to store best fitness of each generation
    plateau_count = 0
    
    for gen in range(max_generations):
        # Calculate fitness for all individuals
        fitness = [compute_fitness(ind, job_operations) for ind in population]
        gen_best_fitness = min(fitness)
        fitness_history.append(gen_best_fitness)  # Store the best fitness of this generation
        
        # Update best solution
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = population[fitness.index(gen_best_fitness)]
            plateau_count = 0
        else:
            plateau_count += 1
        
        # Check for plateau
        if plateau_count >= plateau_threshold:
            print(f"\nPlateau reached at generation {gen}")
            break
        
        # Create new population
        new_population = []
        
        # Apply elitism if enabled
        if elitism:
            elite_idx = fitness.index(min(fitness))
            new_population.append(population[elite_idx])
        
        # Generate rest of the population
        while len(new_population) < population_size:
            # Selection
            if selection_method == "tournament":
                parent1 = tournament_selection(population, fitness)
                parent2 = tournament_selection(population, fitness)
            else:  # rank selection
                parent1, parent2 = rank_selection(population, fitness)
            
            # Crossover
            if random.random() < crossover_rate:
                if crossover_method == "two_point":
                    child1, child2 = two_point_crossover(parent1, parent2)
                else:  # uniform crossover
                    child1, child2 = uniform_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < mutation_rate:
                if mutation_method == "swap":
                    child1 = swap_mutation(child1)
                else:  # inverse mutation
                    child1 = inverse_mutation(child1)
            
            if random.random() < mutation_rate:
                if mutation_method == "swap":
                    child2 = swap_mutation(child2)
                else:  # inverse mutation
                    child2 = inverse_mutation(child2)
            
            new_population.extend([child1, child2])
        
        # Trim population to exact size
        population = new_population[:population_size]
        
        # Print progress
        if gen % 10 == 0:
            print(f"Generation {gen}, Best Fitness: {gen_best_fitness}")
    
    # Calculate final results
    machine_schedules, machine_times, total_processing_time = decode_chromosome(best_solution, job_operations)
    avg_fitness = sum(fitness_history) / len(fitness_history)
    generation_converged = len(fitness_history) - plateau_count
    
    # Return fitness_history along with other results
    return (best_solution, best_fitness, avg_fitness, generation_converged, 
            total_processing_time, max(fitness_history), fitness_history)

def compute_fitness(individual, job_operations):
    _, machine_times, _ = decode_chromosome(individual, job_operations)
    return max(machine_times.values())  # Makespan

def tournament_selection(population, fitness, tournament_size=3):
    """
    Select an individual using tournament selection.
    Returns the winner of the tournament.
    """
    # Randomly select tournament_size individuals
    tournament_indices = random.sample(range(len(population)), tournament_size)
    # Find the one with best fitness (minimum makespan)
    tournament = [(i, fitness[i]) for i in tournament_indices]
    winner_idx = min(tournament, key=lambda x: x[1])[0]
    return population[winner_idx]

def rank_selection(population, fitness):
    """
    Select parents based on their rank in the population.
    Returns two parents.
    """
    # Create list of (individual, fitness) pairs
    pop_fitness = list(zip(population, fitness))
    # Sort by fitness (lower is better)
    sorted_pop = sorted(pop_fitness, key=lambda x: x[1])
    # Extract just the sorted population
    ranked_pop = [ind for ind, _ in sorted_pop]
    
    # Select from the better half of the population
    better_half = len(ranked_pop) // 2
    parent1 = ranked_pop[random.randint(0, better_half)]
    parent2 = ranked_pop[random.randint(0, better_half)]
    
    return parent1, parent2

def two_point_crossover(parent1, parent2):
    """
    Perform two-point crossover between parents.
    Returns two children.
    """
    length = len(parent1)
    # Select two random crossover points
    point1, point2 = sorted(random.sample(range(length), 2))
    
    # Create children by combining segments
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    return child1, child2

def uniform_crossover(parent1, parent2):
    """
    Perform uniform crossover between parents.
    Each gene has 50% chance of being swapped.
    """
    child1, child2 = [], []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)
    return child1, child2

def swap_mutation(individual):
    """
    Perform swap mutation by exchanging two random positions.
    """
    result = individual.copy()
    # Select two random positions
    pos1, pos2 = random.sample(range(len(result)), 2)
    # Swap their values
    result[pos1], result[pos2] = result[pos2], result[pos1]
    return result

def inverse_mutation(individual):
    """
    Perform inverse mutation by reversing a subsequence.
    """
    result = individual.copy()
    # Select two random positions
    pos1, pos2 = sorted(random.sample(range(len(result)), 2))
    # Reverse the subsequence between these positions
    result[pos1:pos2+1] = result[pos1:pos2+1][::-1]
    return result

def plot_fitness_evolution(fitness_history, title, save_path=None):
    """
    Plot the evolution of the minimum total traveling distance (fitness)
    throughout the generations.
    
    Args:
        fitness_history: List of best fitness values for each generation
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', linewidth=2)
    plt.title(title)
    plt.xlabel('Generation')
    plt.ylabel('Minimum Total Traveling Distance')
    plt.grid(True)
    
    # Add markers for initial and final fitness
    plt.plot(0, fitness_history[0], 'ro', label='Initial', markersize=8)
    plt.plot(len(fitness_history)-1, fitness_history[-1], 'go', label='Final', markersize=8)
    
    # Add text annotations for improvement
    initial_fitness = fitness_history[0]
    final_fitness = fitness_history[-1]
    improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100
    
    plt.text(0.02, 0.98, f'Initial: {initial_fitness:.2f}\nFinal: {final_fitness:.2f}\nImprovement: {improvement:.1f}%',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def test_combinations(job_operations):
    """
    Modified test_combinations function to handle the fitness_history
    """
    # Define all the parameter combinations
    selection_methods = ["tournament", "rank"]
    crossover_methods = ["two_point", "uniform"]
    mutation_methods = ["swap", "inverse"]
    elitism_options = [True, False]

    # Create all combinations of the parameters
    combinations = list(itertools.product(selection_methods, crossover_methods, 
                                        mutation_methods, elitism_options))

    # Store results for comparison
    all_results = []

    # Iterate through all combinations and run the genetic algorithm
    for i, (selection_method, crossover_method, mutation_method, elitism) in enumerate(combinations, start=1):
        print(f"\nTesting combination {i}: {selection_method}, {crossover_method}, "
              f"{mutation_method}, Elitism={elitism}")

        # Run genetic algorithm with the current combination of parameters
        (best_solution, best_fitness, avg_fitness, generation_converged,
         total_job_processing_time, max_fitness, fitness_history) = genetic_algorithm(
            job_operations,
            population_size=100,
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            elitism=elitism
        )

        # Store results
        result = {
            'combination': f"Sel:{selection_method}, Cross:{crossover_method}, "
                         f"Mut:{mutation_method}, Elit:{elitism}",
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'generations': generation_converged,
            'total_time': total_job_processing_time,
            'max_fitness': max_fitness
        }
        all_results.append(result)

        # Create plot title
        title = f"Fitness Evolution\n{selection_method.capitalize()} Selection, "
        title += f"{crossover_method.capitalize()} Crossover\n"
        title += f"{mutation_method.capitalize()} Mutation, "
        title += f"Elitism={'On' if elitism else 'Off'}"

        # Plot and save the fitness evolution
        plot_filename = f"fitness_evolution_{i}.png"
        plot_fitness_evolution(fitness_history, title, save_path=plot_filename)

        # Print results
        print(f"Best Fitness: {best_fitness}")
        print(f"Average Fitness: {avg_fitness}")
        print(f"Generations to Converge: {generation_converged}")
        print(f"Total Job Processing Time: {total_job_processing_time}")

    # Print comparison of all results
    print("\nComparison of all combinations:")
    for result in sorted(all_results, key=lambda x: x['best_fitness']):
        print(f"\nCombination: {result['combination']}")
        print(f"Best Fitness: {result['best_fitness']}")
        print(f"Average Fitness: {result['avg_fitness']}")
        print(f"Generations: {result['generations']}")
        print(f"Total Time: {result['total_time']}")

def main():
    # Read problem
    job_operations = read_job_operations_from_file('problem1.txt')
    
    # Test all combinations
    test_combinations(job_operations)

if __name__ == "__main__":
    main()
