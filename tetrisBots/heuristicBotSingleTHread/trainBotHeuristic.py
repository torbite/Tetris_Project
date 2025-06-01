import multiprocessing
from botSingleEntrie import trainingScene
from copy import deepcopy
import time


POPULATION_SIZE = 8
GENERATIONS = 1000

def get_fitness(result):
    return result['score'] + 0.1 * result['steps']

def run_simulation_process(base_weights, label, shared_dict):
    tr = trainingScene()
    tr.weights = deepcopy(base_weights)
    tr.randomizeWeights()

    def update_stats():
        shared_dict['score'] = tr.board.points
        shared_dict['steps'] = tr.totalSteps

    tr.runGame(showBard=False, boardText=label, on_update=update_stats)

    shared_dict['score'] = tr.board.points
    shared_dict['steps'] = tr.totalSteps
    shared_dict['weights'] = tr.weights
    shared_dict['label'] = label

def run_generation(pop_size, base_weights, gen_num, timeout=1000, best_individual=None):
    print(f"Running generation {gen_num} with max {timeout}s...")

    manager = multiprocessing.Manager()
    processes = []
    shared_dicts = []

    # Reserve one slot for best_individual
    num_new_individuals = pop_size - 1

    for i in range(num_new_individuals):
        label = f'GEN {gen_num} IND {i}'
        shared = manager.dict(score=0, steps=0, weights=[], label=label)
        shared_dicts.append(shared)
        p = multiprocessing.Process(target=run_simulation_process, args=(deepcopy(base_weights), label, shared))
        p.start()
        processes.append(p)

    start_time = time.time()
    while time.time() - start_time < timeout:
        if not any(p.is_alive() for p in processes):
            break
        time.sleep(1)

    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()

    results = []
    for shared in shared_dicts:
        results.append({
            'label': shared['label'],
            'score': shared['score'],
            'steps': shared['steps'],
            'weights': shared.get('weights', deepcopy(base_weights))
        })

    # Add best_individual directly
    if best_individual:
        print(f"📦 Carrying over best from last generation: {best_individual['label']}")
        best_copy = {
            'label': f'GEN {gen_num} IND {pop_size - 1}',
            'score': best_individual['score'],
            'steps': best_individual['steps'],
            'weights': deepcopy(best_individual['weights'])
        }
        results.append(best_copy)

    return results

def show_best(weights):
    print("\nShowing best model playing (single process)...\n")
    best = trainingScene()
    best.weights = deepcopy(weights)
    best.runGame(showBard=True, boardText="BEST MODEL")

def evolve(pop_size, generations):
    base_weights = [1.0, 1.0, 1.0, 1.0]
    best_individual = None

    for gen in range(generations):
        print(f"\n=== GENERATION {gen} ===")
        start = time.time()

        population = run_generation(pop_size, base_weights, gen, best_individual=best_individual)

        duration = time.time() - start
        print(f"Generation {gen} completed in {duration:.2f} seconds")

        for p in population:
            print(f"{p['label']} | Score: {p['score']} | Steps: {p['steps']} | Fitness: {get_fitness(p):.2f}")

        best = max(population, key=get_fitness)
        print(f"\n🏆 Best: {best['label']} | Score: {best['score']} | Steps: {best['steps']} | Weights: {best['weights']}")

        show_best(best['weights'])
        base_weights = best['weights']
        best_individual = best

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # <-- adicione `force=True`
    evolve(POPULATION_SIZE, GENERATIONS)