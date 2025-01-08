def read_job_operations_from_file(file_path):
    job_operations = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_job_operations = []
        for line in lines:
            if line.strip():  # Skip empty lines
                operations = list(map(int, line.strip().split()))
                current_job_operations.append(operations)
            else:
                job_operations.append(current_job_operations)
                current_job_operations = []
        if current_job_operations:
            job_operations.append(current_job_operations)
    return job_operations

# Read job operations from a text file
job_operations = read_job_operations_from_file('job_operations.txt')
