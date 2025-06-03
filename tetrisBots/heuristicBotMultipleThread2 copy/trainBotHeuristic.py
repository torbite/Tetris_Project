import multiprocessing
from tetrisBotHeuristic import trainingScene
from copy import deepcopy
import time
import json
import os
import random

POPULATION_SIZE = 8
GENERATIONS = 1000
GAMES_PER_INDIVIDUAL = 5


def get_fitness(result):
    return result['steps']


def crossover(w1, w2):
    return [(a + b) / 2 for a, b in zip(w1, w2)]


def mutate(weights, mutation_rate=0.3, mutation_strength=0.2):
    return [
        w + random.gauss(0, mutation_strength) if random.random() < mutation_rate else w
        for w in weights
    ]


def run_simulation_process_all_games(base_weights, label_prefix, shared_dict, num_games=20):
    game_results = []

    for i in range(num_games):
        tr = trainingScene()
        tr.weights = deepcopy(base_weights)
        tr.randomizeWeights()
        tr.runGame(showBard=False, boardText=f"{label_prefix} [Game {i+1}]")

        result = {
            'label': f"{label_prefix} [Game {i+1}]",
            'score': tr.board.points,
            'steps': tr.totalSteps,
            'weights': deepcopy(tr.weights)
        }
        game_results.append(result)

    shared_dict['games'] = game_results


def run_generation(pop_size, base_weights, gen_num, num_games=5, timeout=1000, best_individual=None):
    print(f"Running generation {gen_num} with {num_games} games per individual...")

    manager = multiprocessing.Manager()
    processes = []
    shared_dicts = []

    parents = []
    if best_individual:
        parents.append(best_individual['weights'])

    while len(parents) < pop_size:
        if len(parents) >= 2:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
        elif len(parents) == 1:
            child = deepcopy(parents[0])
        else:
            child = deepcopy(base_weights)

        child = mutate(child)
        parents.append(child)

    for i, weights in enumerate(parents):
        label = f'GEN {gen_num} IND {i}'
        shared = manager.dict()
        shared_dicts.append(shared)
        p = multiprocessing.Process(target=run_simulation_process_all_games, args=(deepcopy(weights), label, shared, num_games))
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

    all_game_data = []
    individuals = []

    for shared in shared_dicts:
        games = shared.get('games', [])
        if games:
            all_game_data.extend(games)
            avg_steps = sum(g['steps'] for g in games) / len(games)
            individuals.append({
                'label': games[0]['label'].split(' [')[0],
                'score': 0,
                'steps': avg_steps,
                'weights': games[-1]['weights'],
                'games': games
            })

    if individuals:
        best_individual = max(individuals, key=get_fitness)

    return individuals, all_game_data, best_individual


def show_best(weights):
    print("\nShowing best model playing (single process)...\n")
    best = trainingScene()
    best.weights = deepcopy(weights)
    best.runGame(showBard=True, boardText="BEST MODEL")


def evaluate_best_and_save(weights, num_games=10, filename="bests.json"):
    total_score = 0
    total_steps = 0

    for i in range(num_games):
        tr = trainingScene()
        tr.weights = deepcopy(weights)
        tr.runGame(showBard=False, boardText=f"Eval BEST [Game {i+1}]")
        total_score += tr.board.points
        total_steps += tr.totalSteps

    avg_score = total_score / num_games
    avg_steps = total_steps / num_games
    weights_key = str([round(w, 5) for w in weights])

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[weights_key] = [avg_score, avg_steps]

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\U0001F4CA Evaluation saved for weights {weights_key} → Steps: {avg_steps:.2f}")


def evolve(pop_size, generations, games_per_individual):
    base_weights = [20.0,1.0,1.0,1.0,1.0,1.0,10.0]
    best_individual = None

    for gen in range(generations):
        print(f"\n=== GENERATION {gen} ===")
        start = time.time()

        individuals, all_games, best_individual = run_generation(
            pop_size, base_weights, gen, games_per_individual, best_individual=best_individual
        )

        duration = time.time() - start
        print("-" * 90)
        print(f"Generation {gen} completed in {duration:.2f} seconds")

        for p in individuals:
            print(f"{p['label']} | Avg Steps: {p['steps']:.1f} | Fitness: {get_fitness(p):.2f}")

        print(f"\n\U0001F3C6 Best: {best_individual['label']} | Steps: {best_individual['steps']} | Weights: {best_individual['weights']}")
        show_best(best_individual['weights'])
        evaluate_best_and_save(best_individual['weights'], num_games=10)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    evolve(POPULATION_SIZE, GENERATIONS, GAMES_PER_INDIVIDUAL)
