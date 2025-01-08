def read_job_operations_from_file(file_name):
    job_operations = []
    
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()

            for line in lines:
                operations = []
                values = list(map(int, line.split()))  # Read all integers in the line
                # Group values in pairs (machine, processing_time)
                for i in range(0, len(values), 2):
                    operations.append((values[i], values[i+1]))  # (machine, time)
                job_operations.append(operations)

    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    print(f"Job operations loaded: {len(job_operations)} jobs")  # Debug print to check
    return job_operations

# Read job operations from a text file
job_operations = read_job_operations_from_file('problem1.txt')
print(job_operations)
