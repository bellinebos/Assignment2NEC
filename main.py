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
    machine_schedule = [[] for _ in range(len(job_operations[0]))]
    job_indices = [0] * len(job_operations)
    time_tracker = [0] * len(job_operations[0])
    job_end_time = [0] * len(job_operations)

    for gene in chromosome:
        job = gene
        operation_idx = job_indices[job]
        machine, processing_time = job_operations[job][operation_idx]

        start_time = max(time_tracker[machine], job_end_time[job])
        end_time = start_time + processing_time

        machine_schedule[machine].append((start_time, end_time, job))
        time_tracker[machine] = end_time
        job_end_time[job] = end_time
        job_indices[job] += 1

    return machine_schedule

def compute_fitness(chromosome, job_operations):
    decoded_schedule, machine_times = decode_chromosome(chromosome, job_operations)

    # Calculate makespan (maximum time across all machines)
    makespan = max(machine_times.values())
    return makespan

# Read job operations from a text file
job_operations = read_job_operations_from_file('problem1.txt')
print(job_operations)
