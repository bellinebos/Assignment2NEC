import matplotlib.pyplot as plt
import random
import itertools

def read_job_operations_from_file(file_name):
    job_operations = []
    with open(file_name, 'r') as file:
        for line in file:
            # Read values from each line and convert to integers
            values = list(map(int, line.strip().split()))
            # Create tuples of (machine, processing_time) for each operation
            operations = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
            # Append the operations for each job to job_operations
            job_operations.append(operations)
    return job_operations

def initialize_pop(job_operations, population_size):
    population = []
    # Create a list of all job tasks as tuples (job_id, task_id)
    all_job_tasks = [(j, t) for j in range(len(job_operations)) 
                     for t in range(len(job_operations[j]))]
    
    for _ in range(population_size):
        # Create a copy of all job tasks for each individual
        individual = all_job_tasks.copy()
        # Shuffle the tasks randomly
        random.shuffle(individual)
        # Append the shuffled individual to the population
        population.append(individual)
    return population

def decode_chromosome(chromosome, job_operations):
    machine_schedules = {}
    machine_times = {}
    # Initialize job end times to 0 for each job
    job_end_times = {j: 0 for j in range(len(job_operations))}
    
    # Initialize machine schedules and times
    max_machine_id = max(machine for job in job_operations for machine, _ in job)
    for m in range(max_machine_id + 1):
        machine_schedules[m] = []
        machine_times[m] = 0
    
    for job_id, task_id in chromosome:
        machine, processing_time = job_operations[job_id][task_id]
        job_start_time = job_end_times[job_id]
        # Calculate task start time as the maximum of machine availability and job's previous task completion
        task_start_time = max(machine_times[machine], job_start_time)
        # Calculate task end time by adding processing time to start time
        task_end_time = task_start_time + processing_time
        
        # Append task details to machine schedule
        machine_schedules[machine].append({
            'job': job_id,
            'task': task_id,
            'start': task_start_time,
            'end': task_end_time
        })
        # Update machine availability time
        machine_times[machine] = task_end_time
        # Update job completion time
        job_end_times[job_id] = task_end_time
    
    # Return machine schedules, machine times, and total processing time (makespan)
    total_processing_time = max(machine_times.values())
    return machine_schedules, machine_times, total_processing_time

def genetic_algorithm(job_operations, population_size=50, max_generations=5000, 
                     crossover_rate=0.8, mutation_rate=0.05, plateau_threshold=100, 
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
                  child1, child2 = two_point_crossover(parent1, parent2, job_operations)
                else:
                  child1, child2 = uniform_crossover(parent1, parent2, job_operations)

            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < mutation_rate:
              if mutation_method == "swap":
                child1 = swap_mutation(child1, job_operations)
              else:  # inverse mutation
                child1 = inverse_mutation(child1, job_operations)
            
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

def enforce_constraints(individual, job_operations):
    # Constraint 1: No task starts until previous task completed
    job_end_times = {j: 0 for j in range(len(job_operations))}
    # Sort by job and task ID to ensure task order within jobs
    individual_sorted = sorted(individual, key=lambda x: (x[0], x[1]))
    
    result = []
    # Constraint 2: Machine can only work on one task at a time
    machine_times = {m: 0 for m in range(len(job_operations))}
    
    for job_id, task_id in individual_sorted:
        machine, processing_time = job_operations[job_id][task_id]
        
        job_start_time = job_end_times[job_id]
        # Takes maximum of machine availability and job's previous task completion
        task_start_time = max(machine_times[machine], job_start_time)
        
        # Constraint 3: Task must run to completion
        task_end_time = task_start_time + processing_time
        
        result.append((job_id, task_id))
        job_end_times[job_id] = task_end_time  # Update job completion time
        machine_times[machine] = task_end_time  # Update machine availability
    return result

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

def two_point_crossover(parent1, parent2, job_operations):
    length = len(parent1)
    point1, point2 = sorted(random.sample(range(length), 2))
    
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    # Add constraint enforcement
    child1 = enforce_constraints(child1, job_operations)
    child2 = enforce_constraints(child2, job_operations)
    
    return child1, child2

def uniform_crossover(parent1, parent2, job_operations):
    """
    Perform uniform crossover between two parents to produce two children.
    """
    # Initialize empty children
    child1, child2 = [], []

    # Iterate through the genes and perform uniform crossover
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])

    # Ensure constraints with the job operations
    child1 = enforce_constraints(child1, job_operations)
    child2 = enforce_constraints(child2, job_operations)

    return child1, child2

def swap_mutation(individual, job_operations):
    """
    Perform swap mutation by exchanging two random positions.
    """
    result = individual.copy()
    # Select two random positions
    pos1, pos2 = random.sample(range(len(result)), 2)
    # Swap their values
    result[pos1], result[pos2] = result[pos2], result[pos1]
    
    # Ensure constraints:
    result = enforce_constraints(result, job_operations)
    
    return result

def inverse_mutation(individual, job_operations):
    """
    Perform inverse mutation on the given individual.
    This function inverts a section of the individual's schedule.
    """
    # Pick two random positions to swap
    start_idx, end_idx = sorted(random.sample(range(len(individual)), 2))

    # Reverse the section of the individual's schedule
    individual[start_idx:end_idx+1] = reversed(individual[start_idx:end_idx+1])

    # Enforce constraints with the updated individual
    return enforce_constraints(individual, job_operations)

def plot_fitness_evolution(fitness_history, title, save_path=None):
    """
    Plot the evolution of the minimum total traveling distance (fitness)
    throughout the generations.
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
            population_size=200,    # adjust population size according to problem: 200 for problem 1, 400 for problem 2 and 600 for problem 3
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
        title += f"{crossover_method.capitalize()} Crossover, {mutation_method.capitalize()} Mutation"
        
        # Plot fitness evolution for this combination
        plot_fitness_evolution(fitness_history, title)

    # Return the results of all combinations tested
    return all_results

def main():
    # Read the job operations from the file (adjust the file path as needed)
    job_operations = read_job_operations_from_file('problem1.txt') #adjust txt file to problem1.txt, problem2.txt or problmem3.txt
    
    # Test all combinations of selection, crossover, mutation methods, and elitism options
    results = test_combinations(job_operations)
    
    # Print the results for all combinations tested
    print("\nComparison of all combinations:")
    for result in sorted(results, key=lambda x: x['best_fitness']):
        print(f"\nCombination: {result['combination']}")
        print(f"Best Fitness: {result['best_fitness']}")
        print(f"Average Fitness: {result['avg_fitness']}")
        print(f"Generations: {result['generations']}")
        print(f"Total Time: {result['total_time']}")

if __name__ == "__main__":
    main()
