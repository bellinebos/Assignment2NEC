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

# Read job operations from a text file
job_operations = read_job_operations_from_file('problem1.txt')
print(job_operations)
