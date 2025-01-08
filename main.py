def read_job_operations_from_file(file_name):
    job_operations = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        
        num_jobs = len(lines) // 2  # Since we have two parts for each job
        for i in range(num_jobs):
            line1 = list(map(int, lines[i * 2].split()))
            line2 = list(map(int, lines[i * 2 + 1].split()))
            
            job_operations.append(list(zip(line1[::2], line1[1::2])))  # Machine, Processing time pairs for the first line
            job_operations.append(list(zip(line2[::2], line2[1::2])))  # Machine, Processing time pairs for the second line
            
    return job_operations

# Read job operations from a text file
job_operations = read_job_operations_from_file('problem1.txt')
print(job_operations)
