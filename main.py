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
    
    total_processing_time = max(machine_times.values())
    return machine_schedules, machine_times, total_processing_time

def genetic_algorithm(job_operations, population_size=100, max_generations=500,
                     crossover_rate=0.8, mutation_rate=0.05, plateau_threshold=50,
                     selection_method="tournament", crossover_method="two_point",
                     mutation_method="swap", elitism=True):

    population = initialize_pop(job_operations, population_size)
    best_fitness = float('inf')
    best_solution = None
    fitness_history = []
    plateau_count = 0
    
    for gen in range(max_generations):
        fitness = [compute_fitness(ind, job_operations) for ind in population]
        gen_best_fitness = min(fitness)
        fitness_history.append(gen_best_fitness)
        
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness  
            best_solution = population[fitness.index(gen_best_fitness)]
            plateau_count = 0
        else:
            plateau_count += 1
        
        if plateau_count >= plateau_threshold:
            break
            
        new_population = []
        
        if elitism:  
            elite_idx = fitness.index(min(fitness))
            new_population.append(population[elite_idx])
        
        while len(new_population) < population_size:
            if selection_method == "tournament":
                parent1 = tournament_selection(population, fitness)
                parent2 = tournament_selection(population, fitness)
            else:
                parent1, parent2 = rank_selection(population, fitness)
            
            if random.random() < crossover_rate:
                if crossover_method == "two_point":
                    child1, child2 = feasible_two_point_crossover(parent1, parent2, job_operations) 
                else:
                    child1, child2 = feasible_uniform_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            if random.random() < mutation_rate:
                if mutation_method == "swap":
                    child1 = feasible_swap_mutation(child1, job_operations)
                else:
                    child1 = feasible_inverse_mutation(child1, job_operations)
            
            if random.random() < mutation_rate:
                if mutation_method == "swap":
                    child2 = feasible_swap_mutation(child2, job_operations)
                else:
                    child2 = feasible_inverse_mutation(child2, job_operations)
            
            new_population.extend([child1, child2])
        
        population = new_population[:population_size]
        
    machine_schedules, machine_times, total_processing_time = decode_chromosome(best_solution, job_operations)
    avg_fitness = sum(fitness_history) / len(fitness_history) 
    generation_converged = len(fitness_history) - plateau_count
    
    return (best_solution, best_fitness, avg_fitness, generation_converged,
            total_processing_time, max(fitness_history), fitness_history)

def compute_fitness(individual, job_operations):
    _, machine_times, _ = decode_chromosome(individual, job_operations)
    return max(machine_times.values())

def tournament_selection(population, fitness, tournament_size=3):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament = [(i, fitness[i]) for i in tournament_indices]
    winner_idx = min(tournament, key=lambda x: x[1])[0]
    return population[winner_idx]

def rank_selection(population, fitness):
    pop_fitness = list(zip(population, fitness))
    sorted_pop = sorted(pop_fitness, key=lambda x: x[1])
    ranked_pop = [ind for ind, _ in sorted_pop]
    
    better_half = len(ranked_pop) // 2
    parent1 = ranked_pop[random.randint(0, better_half)]
    parent2 = ranked_pop[random.randint(0, better_half)]
    
    return parent1, parent2

def feasible_two_point_crossover(parent1, parent2, job_operations):
    n = len(parent1)
    point1, point2 = 0, 0

    while point1 == point2:
        point1, point2 = sorted(random.sample(range(1, n), 2))
        for ops in job_operations:
            if any(t > point1 and t < point2 for _, t in ops):
                point1, point2 = 0, 0
                break

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:] 
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    
    return child1, child2

def feasible_uniform_crossover(parent1, parent2):
    n = len(parent1)
    child1, child2 = [-1] * n, [-1] * n
    
    for i in range(n):
        if random.random() < 0.5:
            child1[i], child2[i] = parent1[i], parent2[i]
        else:
            child1[i], child2[i] = parent2[i], parent1[i]
    
    remaining1 = [task for task in parent1 if task not in child1]
    remaining2 = [task for task in parent2 if task not in child2]
    
    for i in range(n):
        if child1[i] == -1:
            child1[i] = remaining2.pop(0)
        if child2[i] == -1:
            child2[i] = remaining1.pop(0)
            
    return child1, child2

def feasible_swap_mutation(individual, job_operations):
    n = len(individual)
    pos1, pos2 = 0, 0

    while pos1 == pos2 or individual[pos1][0] == individual[pos2][0]:
        pos1, pos2 = random.sample(range(n), 2)

    job1, task1 = individual[pos1]
    job2, task2 = individual[pos2]
    ops1, ops2 = job_operations[job1], job_operations[job2]
    
    if ((ops1[task1-1][1] if task1 > 0 else -1) > pos2 or
    (ops1[task1+1][1] if task1 < len(ops1)-1 else float('inf')) < pos2 or
    (ops2[task2-1][1] if task2 > 0 else -1) > pos1 or 
    (ops2[task2+1][1] if task2 < len(ops2)-1 else float('inf')) < pos1):
        return individual

    result = individual.copy()
    result[pos1], result[pos2] = result[pos2], result[pos1]  
    return result

def feasible_inverse_mutation(individual, job_operations):
    n = len(individual)
    pos1, pos2 = 0, 0

    while pos1 == pos2:
        pos1, pos2 = sorted(random.sample(range(n), 2))
        for ops in job_operations:
            if any(t > pos1 and t < pos2 for _, t in ops):
                pos1, pos2 = 0, 0
                break

    result = individual.copy()
    result[pos1:pos2+1] = result[pos1:pos2+1][::-1]
    return result

def plot_fitness_evolution(fitness_history, title, num_runs, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', linewidth=2)
    plt.title(f"{title}\nAveraged over {num_runs} runs")
    plt.xlabel('Generation')
    plt.ylabel('Minimum Makespan')
    plt.grid(True)
    
    initial_fitness = fitness_history[0]
    final_fitness = fitness_history[-1]
    improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100
    
    plt.text(0.02, 0.98, f'Initial: {initial_fitness:.2f}\nFinal: {final_fitness:.2f}\nImprovement: {improvement:.1f}%',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path)
        
    plt.close()

def test_combinations(job_operations, num_runs=30):
    selection_methods = ["tournament", "rank"]
    crossover_methods = ["two_point", "uniform"]
    mutation_methods = ["swap", "inverse"] 
    elitism_options = [True, False]

    combinations = list(itertools.product(selection_methods, crossover_methods, 
                                           mutation_methods, elitism_options))

    all_results = []

    for i, (selection_method, crossover_method, mutation_method, elitism) in enumerate(combinations, start=1):
        print(f"\nTesting combination {i}: {selection_method}, {crossover_method}, "
              f"{mutation_method}, Elitism={elitism}")

        run_results = []
        for run in range(num_runs):
            result = genetic_algorithm(
                job_operations,
                population_size=100,
                selection_method=selection_method,
                crossover_method=crossover_method, 
                mutation_method=mutation_method,
                elitism=elitism
            )
            run_results.append(result)

        avg_best_fitness = sum(r[1] for r in run_results) / num_runs
        avg_gen_converged = sum(r[3] for r in run_results) / num_runs
        avg_total_time = sum(r[4] for r in run_results) / num_runs
        
        print(f"Average Best Fitness: {avg_best_fitness:.2f}")
        print(f"Average Generations: {avg_gen_converged:.2f}")
        print(f"Average Total Time: {avg_total_time:.2f}")
        
        title = f"Fitness Evolution\n{selection_method.capitalize()} Selection, "
        title += f"{crossover_method.capitalize()} Crossover\n" 
        title += f"{mutation_method.capitalize()} Mutation, "
        title += f"Elitism={'On' if elitism else 'Off'}"
        
        plot_filename = f"fitness_evolution_combination_{i}.png"
        plot_fitness_evolution(run_results[-1][-1], title, num_runs, save_path=plot_filename)
        
        result = {
            'combination': f"Sel:{selection_method}, Cross:{crossover_method}, "
                         f"Mut:{mutation_method}, Elit:{elitism}",
            'avg_best_fitness': avg_best_fitness,
            'avg_generations': avg_gen_converged,
            'avg_total_time': avg_total_time 
        }
        all_results.append(result)

    print("\nComparison of all combinations:")
    for result in sorted(all_results, key=lambda x: x['avg_best_fitness']):
        print(f"\nCombination: {result['combination']}")
        print(f"Average Best Fitness: {result['avg_best_fitness']:.2f}")
        print(f"Average Generations: {result['avg_generations']:.2f}") 
        print(f"Average Total Time: {result['avg_total_time']:.2f}")

def main():
    job_operations = read_job_operations_from_file('problem1.txt')
    test_combinations(job_operations, num_runs=30)

if __name__ == "__main__":
    main()
