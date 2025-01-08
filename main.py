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

# Read job operations from a text file
job_operations = read_job_operations_from_file('problem1.txt')
print(job_operations)
