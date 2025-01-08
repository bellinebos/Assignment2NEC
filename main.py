import numpy as np
import matplotlib.pyplot as plt
import random

def read_job_operations_from_file(file_name):
    job_operations = []
    
    with open(file_name, 'r') as file:
        lines = file.readlines()

        for line in lines:
            operations = []
            values = list(map(int, line.split()))  # Read all integers in the line
            # Group values in pairs (machine, processing_time)
            for i in range(0, len(values), 2):
                operations.append((values[i], values[i+1]))  # (machine, time)
            job_operations.append(operations)

    return job_operations

def generate_chromosome(job_operations):
    chromosome = []
    for job in range(len(job_operations)):
        # Assuming each job has multiple operations
        num_operations = len(job_operations[job])
        operation_idx = random.randint(0, num_operations - 1)
        chromosome.append((job, operation_idx))  # Proper tuple (job, operation_idx)
    return chromosome

def generate_initial_population(num_jobs, num_tasks, population_size):
    population = []
    for _ in range(population_size):
        individual = []
        for job_id in range(num_jobs):
            individual += [(job_id, task_id) for task_id in range(num_tasks)]  # Include job-task pairing
        random.shuffle(individual)  # Shuffle the sequence to create diverse individuals
        population.append(individual)
    return population

def decode_chromosome(chromosome, job_operations):
    machine_schedules = {machine: [] for machine in range(len(job_operations[0]))}  # Machine schedules
    machine_times = {machine: 0 for machine in range(len(job_operations[0]))}  # Machine times
    job_end_times = {job_id: 0 for job_id in range(len(job_operations))}  # End time of the last task in each job

    for job_id, task_id in chromosome:
        machine, processing_time = job_operations[job_id][task_id]

        # Ensure that no task starts before the previous task of the same job is completed
        job_start_time = job_end_times[job_id]  # The task must start after the last task of the same job finishes
        task_start_time = max(machine_times[machine], job_start_time)  # The task starts after the machine is available and the job's precedence constraint is satisfied
        task_end_time = task_start_time + processing_time  # Task's end time

        # Assign task to the machine
        machine_schedules[machine].append((job_id, task_id, task_start_time, task_end_time))

        # Update machine's availability time and job's end time
        machine_times[machine] = task_end_time
        job_end_times[job_id] = task_end_time  # Update the job's last task end time

    return machine_schedules, machine_times

def repair_chromosome(chromosome):
    """
    Ensures that each job-task pair appears exactly once in the chromosome.
    """
    seen = set()
    repaired = []

    for task in chromosome:
        if task not in seen:
            repaired.append(task)
            seen.add(task)

    return repaired

def genetic_algorithm(job_operations, population_size=100, generations=500, crossover_rate=0.8, mutation_rate=0.01, elitism=True,
                      selection_method="rank", crossover_method="two_point", mutation_method="swap"):

    print(f"Running with Combination: Selection={selection_method}, Crossover={crossover_method}, Mutation={mutation_method}, Elitism={elitism}")

    num_jobs = len(job_operations)
    num_tasks = max(len(job) for job in job_operations)  # Maximum tasks across jobs
    population = generate_initial_population(num_jobs, num_tasks, population_size)

    best_fitness = float('inf')
    best_solution = None
    fitness_history = []
    avg_fitness_history = []
    generation_converged = 0

    for generation in range(generations):
        fitness = [compute_fitness(individual, job_operations) for individual in population]
        
        # Track the best fitness
        generation_best_fitness = min(fitness)
        fitness_history.append(generation_best_fitness)

        avg_fitness = np.mean(fitness)
        avg_fitness_history.append(avg_fitness)

        if generation > 0 and generation_best_fitness == fitness_history[generation - 1]:
            generation_converged += 1
        else:
            generation_converged = 0

        # Elitism: Keep the best individual
        new_population = []
        if elitism:
            elite_idx = np.argmin(fitness)
            new_population.append(population[elite_idx])

        while len(new_population) < population_size:
            if selection_method == "rank":
                parent1 = rank_selection(population, fitness)
                parent2 = rank_selection(population, fitness)
            elif selection_method == "tournament":
                parent1 = tournament_selection(population, fitness)
                parent2 = tournament_selection(population, fitness)

            if crossover_method == "two_point":
                if random.random() < crossover_rate:
                    child1, child2 = two_point_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
            elif crossover_method == "uniform":
                if random.random() < crossover_rate:
                    child1, child2 = uniform_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]

            if mutation_method == "swap":
                if random.random() < mutation_rate:
                    swap_mutation(child1)
                if random.random() < mutation_rate:
                    swap_mutation(child2)
            elif mutation_method == "inverse":
                if random.random() < mutation_rate:
                    inverse_mutation(child1)
                if random.random() < mutation_rate:
                    inverse_mutation(child2)

            new_population.extend([repair_chromosome(child1), repair_chromosome(child2)])

        # Update the population
        population = new_population[:population_size]

        if generation_best_fitness < best_fitness:
            best_fitness = generation_best_fitness
            best_solution = population[np.argmin(fitness)]

        print(f"Generation {generation + 1}: Best Fitness = {generation_best_fitness}, Average Fitness = {avg_fitness}")

        if generation_converged > 50:
            print(f"Convergence detected at generation {generation + 1}")
            break

    # Decode the best solution to get the schedule and machine times
    machine_schedules, machine_times = decode_chromosome(best_solution, job_operations)
    
    # Calculate the processing time (makespan) of the best solution
    processing_time = max(machine_times.values())

    print(f"Best Fitness: {best_fitness}")
    print(f"Average Fitness (last generation): {avg_fitness}")
    print(f"Generations to Converge: {generation_converged}")
    print(f"Processing Time (Makespan) of Best Solution: {processing_time}")

    # Returning the solution, fitness, processing time, etc.
    return best_solution, best_fitness, fitness_history, avg_fitness_history, generation_converged, processing_time

def compute_fitness(individual, job_operations):
    """
    Calculate fitness as the makespan of the schedule.
    """
    machine_times = {}  # Track end times for each machine

    for job_id, task_id in individual:
        machine, processing_time = job_operations[job_id][task_id]

        # Determine the earliest start time for the machine
        machine_start_time = machine_times.get(machine, 0)
        machine_times[machine] = machine_start_time + processing_time

    # Fitness is the maximum end time (makespan)
    return max(machine_times.values())

def rank_selection(population, fitness):
    """
    Perform rank-based selection.
    Args:
        population (list): The current population.
        fitness (list): Fitness values of the population.
    Returns:
        Selected individual.
    """
    sorted_population = [x for _, x in sorted(zip(fitness, population))]
    total_rank = sum(range(1, len(sorted_population) + 1))
    probabilities = [rank / total_rank for rank in range(1, len(sorted_population) + 1)]
    selected_index = random.choices(range(len(sorted_population)), weights=probabilities, k=1)[0]
    return sorted_population[selected_index]

def tournament_selection(population, fitness, tournament_size=3):
    """
    Perform tournament selection.
    Args:
        population (list): The current population.
        fitness (list): Fitness values of the population.
        tournament_size (int): Number of individuals in the tournament.
    Returns:
        Selected individual.
    """
    selected = random.sample(list(zip(population, fitness)), tournament_size)
    selected.sort(key=lambda x: x[1])  # Sort by fitness (lower is better)
    return selected[0][0]  # Return the best individual

import random

def uniform_crossover(parent1, parent2):
    """
    Perform uniform crossover between two parent chromosomes.

    Args:
    - parent1: List of genes representing the first parent.
    - parent2: List of genes representing the second parent.

    Returns:
    - offspring: List of genes representing the offspring after crossover.
    """
    offspring = []
    
    # Iterate through each gene in the chromosomes
    for gene1, gene2 in zip(parent1, parent2):
        # Randomly select a gene from either parent
        if random.random() < 0.5:  # 50% chance to pick from parent1 or parent2
            offspring.append(gene1)
        else:
            offspring.append(gene2)
    
    return offspring

def two_point_crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    # Ensure valid job-task sequences
    child1 = repair_chromosome(child1)
    child2 = repair_chromosome(child2)

    return child1, child2

def swap_mutation(individual):
    """
    Perform swap mutation on an individual.
    Args:
        individual (list): Individual to mutate.
    Returns:
        Mutated individual.
    """
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

def inverse_mutation(individual):
    """
    Perform inverse mutation on an individual.
    Args:
        individual (list): Individual to mutate.
    Returns:
        Mutated individual.
    """
    idx1, idx2 = sorted(random.sample(range(len(individual)), 2))
    individual[idx1:idx2] = reversed(individual[idx1:idx2])

def plot_fitness_history(fitness_history):
    plt.plot(fitness_history)
    plt.title("Fitness Evolution")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.show()

# Read job operations from a text file
job_operations = read_job_operations_from_file('problem1.txt')

# Combination 1

best_solution, best_fitness, fitness_history, avg_fitness_history, generation_converged, computational_time = genetic_algorithm(
    job_operations,
    population_size=200,
    generations=500,
    crossover_rate=0.8,
    mutation_rate=0.05,
    elitism=False,
    selection_method="rank",
    crossover_method="two_point",
    mutation_method="swap"
)

plot_fitness_history(fitness_history)
