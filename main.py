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

def generate_initial_population(num_jobs, num_machines, population_size):
    population = []
    for _ in range(population_size):
        individual = []
        for job_id in range(num_jobs):
            individual += [job_id] * num_machines  # Assuming each job has 'num_machines' operations
        random.shuffle(individual)
        population.append(individual)
    return population

def decode_chromosome(chromosome, job_operations):
    decoded_schedule = []
    machine_times = {i: 0 for i in range(len(job_operations[0]))}  # Initialize machine_times for all machines

    for i, job_idx in enumerate(chromosome):
        job_ops = job_operations[job_idx]
        op_idx = i % len(job_ops)  # Select operation based on chromosome position
        operation = job_ops[op_idx]  # Get the operation (list of tuples: machine, time)

        # Check if operation is valid (should be a tuple (machine, processing time))
        if isinstance(operation, tuple):
            machine, processing_time = operation
            decoded_schedule.append((job_idx, op_idx, machine, processing_time))
            machine_times[machine] += processing_time  # Add the processing time to the machine

    return decoded_schedule, machine_times

def genetic_algorithm(job_operations, population_size=100, generations=500, crossover_rate=0.8, mutation_rate=0.2, elitism=True,
                      selection_method="rank", crossover_method="two_point", mutation_method="swap"):

    # Print the current combination being used
    print(f"Running with Combination: Selection={selection_method}, Crossover={crossover_method}, Mutation={mutation_method}, Elitism={elitism}")

    num_jobs = len(job_operations)
    num_machines = len(job_operations[0])
    population = generate_initial_population(num_jobs, num_machines, population_size)

    best_fitness = float('inf')
    best_solution = None
    fitness_history = []

    for generation in range(generations):
        fitness = [compute_fitness(individual, job_operations) for individual in population]
        new_population = []

        # Elitism: Keep the best individual from the current generation
        if elitism:
            elite_idx = np.argmin(fitness)
            new_population.append(population[elite_idx])

        while len(new_population) < population_size:
            # Select parents using the specified selection method
            if selection_method == "rank":
                parent1 = rank_selection(population, fitness)
                parent2 = tournament_selection(population, fitness)  # Alternatively, you can also use rank_selection for parent2
            elif selection_method == "tournament":
                parent1 = tournament_selection(population, fitness)
                parent2 = tournament_selection(population, fitness)

            # Apply the specified crossover method
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

            # Apply the specified mutation method
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

            # Add the children to the new population
            new_population.extend([child1, child2])

        # Limit the new population to the specified population size
        population = new_population[:population_size]

        # Track the best solution of the current generation
        generation_best_fitness = min(fitness)
        fitness_history.append(generation_best_fitness)

        if generation_best_fitness < best_fitness:
            best_fitness = generation_best_fitness
            best_solution = population[np.argmin(fitness)]

        print(f"Generation {generation + 1}: Best Fitness = {generation_best_fitness}")

    return best_solution, best_fitness, fitness_history

def compute_fitness(chromosome, job_operations):
    decoded_schedule, machine_times = decode_chromosome(chromosome, job_operations)

    # Calculate makespan (maximum time across all machines)
    makespan = max(machine_times.values())
    return makespan

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

def two_point_crossover(parent1, parent2):
    """
    Perform two-point crossover.
    Args:
        parent1 (list): First parent.
        parent2 (list): Second parent.
    Returns:
        Two offspring.
    """
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
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
print(job_operations)
