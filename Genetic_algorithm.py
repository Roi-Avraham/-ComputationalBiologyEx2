import random
import string
from statistics import mean
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk

enc_path = ""
# define parameters for the genetic algorithm
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
MUTATION_RATE = 0.9
ELITE_SIZE = 10
TOURNAMENT_SIZE = 30
# the mode of the algorithm - C for classic D for darvin L for lamark
MODE = "C"


def get_input():
    global enc_path
    global MODE
    MODE = entry_mode.get()
    enc_path = entry_path.get()
    root.destroy()


root = tk.Tk()

mode_var = tk.StringVar(value="C")
entry_path_var = tk.StringVar(value="enc.txt")

label_mode = tk.Label(root, text="Enter the mode run (C for classic, D for darvin and L for lamark):")
label_mode.pack()

entry_mode = tk.Entry(root, textvariable=mode_var)
entry_mode.pack()

label_path = tk.Label(root, text="Enter the path of the encrypted file:")
label_path.pack()

entry_path = tk.Entry(root, textvariable=entry_path_var)
entry_path.pack()

button = tk.Button(root, text="Submit", command=get_input)
button.pack()

width = 350
height = 100
root.geometry(f"{width}x{height}")

root.mainloop()

if MODE != "C" and MODE != "D" and MODE != "L":
    print("the mode run must be C (for classic) or D (for darwin) or L (for lamark)")
    exit(-1)

print("mode: ", MODE)

# read in the ciphertext
with open(enc_path, 'r') as f:
    ciphertext = f.read().strip().lower()
    ciphertext = ciphertext.translate(str.maketrans("", "", string.punctuation))

# read in the dictionary file
with open('dict.txt', 'r') as f:
    dictionary = set(word.strip().lower() for word in f.readlines())

# read in the letter frequency files
with open('Letter_Freq.txt', 'r') as f:
    letter_freq = {letter: float(freq) for freq, letter in [line.strip().split() for line in f.readlines()]}
letter_pair_freq = {}
with open('Letter2_Freq.txt', 'r') as f:
    for line in f:
        try:
            freq, pair = line.strip().split()
            letter_pair_freq[pair] = float(freq)
        except ValueError:
            continue

# the number of times of calculation fittness in the algorithm
steps = 0

# list of the best solution and its fittness in each run (in case of early settlement)
best_scores = []

# indicates the algorithm get to the optimal solution (when all the words we deciphered are show in dict).
STOP = False

# data for the graphs
dict_graph = {}


# define fitness function
def calculate_fitness(plaintext):
    """
        calculate_fitness calculate the fittness of the solution that get to the plaintext
        :arg plaintext - the text the current solution deciphered
        :return the fittness
    """

    global steps
    global STOP
    steps += 1
    plaintext_freq = {}
    plaintext_pair_freq = {}
    plaintext_words = set(plaintext.split())
    num_words_in_dict = len(plaintext_words.intersection(dictionary))
    if num_words_in_dict == len(plaintext_words):
        STOP = True
    count_alpha = 0
    for i in range(len(plaintext)):
        letter = plaintext[i]
        if letter.isalpha():
            count_alpha += 1
            plaintext_freq[letter] = plaintext_freq.get(letter, 0) + 1
            if i < len(plaintext) - 1 and plaintext[i + 1].isalpha():
                pair = plaintext[i:i + 2]
                plaintext_pair_freq[pair] = plaintext_pair_freq.get(pair, 0) + 1
    letter_fitness = sum(
        1 / (1 + abs(plaintext_freq.get(letter, 0) / count_alpha - letter_freq.get(letter, 0))) for letter in
        letter_freq)
    pair_fitness = sum(
        1 / (1 + abs(plaintext_pair_freq.get(pair, 0) / count_alpha - letter_pair_freq.get(pair, 0))) for pair in
        letter_pair_freq)
    return 2 * num_words_in_dict + letter_fitness + pair_fitness


def new_word(word):
    """
            new_word gets word and replace duplicates letters with random non-appeared letters
            :arg word - some word
            :return the word but with the replaced letters
    """

    char_counts = {}
    for c in word:
        char_counts[c] = char_counts.get(c, 0) + 1

    duplicates = [c for c, count in char_counts.items() if count > 1]
    non_appear = [c for c in 'abcdefghijklmnopqrstuvwxyz' if c not in char_counts]

    # Replace duplicates with random non-appeared letters
    for char in duplicates:
        char_not_appear = random.choice(non_appear)
        word = word.replace(char, char_not_appear, 1)
        # remove chosen letter from non-appear list
        non_appear.remove(char_not_appear)

    return word


# define crossover operator
def crossover(parent1, parent2):
    """
        crossover function that 2 solutions (parent 1 and parent 2) and creates 2 children.
        first the function random index of cut off and after that it creates the 2 children in this way:
        the first child will be the parent1 until cut off index chained the parent2 from fut off index until the end.
        the second child will be the parent2 until cut off index chained the parent1 from fut off index until the end.
        after that replace duplicates letters with random non-appeared letters to the both children by calling new_word
         function
        :param parent1 - solution number 1
        :param parent2 - solution number 2
        :return the children
    """

    cutoff = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cutoff] + parent2[cutoff:]
    child2 = parent2[:cutoff] + parent1[cutoff:]
    child1 = new_word(child1)
    child2 = new_word(child2)
    return child1, child2


def mutate(individual):
    """
    Mutate a candidate permutation by swapping two random elements.
    """

    if random.random() < MUTATION_RATE:
        candidate = list(individual)
        pos1, pos2 = random.sample(range(26), 2)
        candidate[pos1], candidate[pos2] = candidate[pos2], candidate[pos1]
        return ''.join(candidate)
    return individual


def local_opt(individual, old_fittness):
    """
        Mutate n (random int) times an individual permutation by swapping two random elements.
        and accept this mutation only if the new fittness is bigger then the old one.
    """

    n = random.randint(1, 2)
    for i in range(n):
        mutated = list(individual)
        index1 = random.randint(0, len(mutated) - 1)
        index2 = index1
        while index1 == index2:
            index2 = random.randint(0, len(mutated) - 1)
        mutated[index1], mutated[index2] = mutated[index2], mutated[index1]
        new_individual = ''.join(mutated)
        new_fittness = calculate_fitness(
            ciphertext.translate(str.maketrans(new_individual, 'abcdefghijklmnopqrstuvwxyz')))
        if new_fittness > old_fittness:
            old_fittness = new_fittness
            individual = new_individual

    return individual, old_fittness


def are_last_n_close(lst):
    """
        checks if the diffr between the last 10 elements of lst are at most 1.
    """

    n = len(lst)
    if n < 10:
        return False

    last_10 = lst[-10:]
    return all(abs(last_10[i] - last_10[i - 1]) <= 1 for i in range(1, 10))


def enter_to_dict(index, average_fittness_array, best_fitness_array, steps_array, worst_fittness_array):
    """
       enter all the data we need for the graph to dict_graph
    """

    global dict_graph
    data = [average_fittness_array, best_fitness_array, steps_array, worst_fittness_array]
    dict_graph[index] = data


def gentic_algorithm(index):
    """
       Runs a genetic algorithm to optimize a population of individuals.

       Args:
           index (int): The index of the algorithm run.

       Returns:
           bool: True if the algorithm completes successfully, False otherwise.
    """
    global dict_graph, STOP

    # generate initial population
    population = [''.join(random.sample('abcdefghijklmnopqrstuvwxyz', 26)) for i in range(POPULATION_SIZE)]

    best_fitness = float('-inf')
    average_fittness_array = []
    best_fitness_array = []
    worst_fittness_array = []
    steps_array = []

    for generation in range(1, NUM_GENERATIONS + 1):
        fitnesses = []
        if MODE == "C":
            # calculate fitness for each individual
            fitnesses = [
                (individual,
                 calculate_fitness(ciphertext.translate(str.maketrans(individual, 'abcdefghijklmnopqrstuvwxyz')))) for
                individual in population]

        elif MODE == "L":
            # calculate fitness for each individual
            new_population = []
            for individual in population:
                new_individual, new_fittness = local_opt(individual, calculate_fitness(
                    ciphertext.translate(str.maketrans(individual, 'abcdefghijklmnopqrstuvwxyz'))))
                fitnesses.append((new_individual, new_fittness))
                new_population.append(new_individual)

        elif MODE == "D":
            for individual in population:
                new_individual, new_fittness = local_opt(individual, calculate_fitness(
                    ciphertext.translate(str.maketrans(individual, 'abcdefghijklmnopqrstuvwxyz'))))
                fitnesses.append((individual, new_fittness))

        fitnesses.sort(key=lambda x: x[1], reverse=True)

        if fitnesses[0][1] > best_fitness:
            best_individual, best_fitness = fitnesses[0]

        best_fitness_array.append(best_fitness)
        worst_fittness_array.append(fitnesses[-1][1])

        # save the average fittness
        fit = [fitnesse[1] for fitnesse in fitnesses]
        average = mean(fit)
        average_fittness_array.append(average)

        global steps
        steps_array.append(steps)

        if STOP:
            enter_to_dict(index, average_fittness_array, best_fitness_array, steps_array, worst_fittness_array)
            break

        if are_last_n_close(average_fittness_array):
            best_scores.append(fitnesses[0])
            print("Early convergence - starting again")
            enter_to_dict(index, average_fittness_array, best_fitness_array, steps_array, worst_fittness_array)
            return False

        # select elite individuals
        elite = [individual for individual, fitness in fitnesses[:ELITE_SIZE]]

        # select parents via tournament selection
        parents = []
        weights = [pair[1] for pair in fitnesses]
        total_weight = sum(weights)
        probability_dist = [w / total_weight for w in weights]

        for i in range(POPULATION_SIZE - ELITE_SIZE):
            # randomly select from the pairs using the probability distribution
            tournament_fitnesses = random.choices(fitnesses, weights=probability_dist, k=TOURNAMENT_SIZE)
            tournament_fitnesses.sort(key=lambda x: x[1], reverse=True)
            parents.append(tournament_fitnesses[0][0])

        # breed offspring via crossover
        offspring = []
        for i in range(0, POPULATION_SIZE - ELITE_SIZE, 2):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)

        # apply mutation to offspring
        population = elite + [mutate(individual) for individual in offspring]
        print(f"Generation {generation} - Steps until now: {steps}, Best Fitness: {best_fitness}")

    best_scores.append(fitnesses[0])
    enter_to_dict(index, average_fittness_array, best_fitness_array, steps_array, worst_fittness_array)
    STOP = True
    return True


def create_graph(max_index):
    """
       Creates and displays a graph based on the data stored in the 'dict_graph' global variable.

       Args:
           max_index (int): The index of the data in 'dict_graph' to be plotted.

       Returns:
           None
    """

    global dict_graph
    avg = dict_graph[max_index][0]
    best = dict_graph[max_index][1]
    worst = dict_graph[max_index][3]
    generation = [i for i in range(1, len(avg) + 1)]
    generation = np.array(generation)

    plt.figure()  # create a new figure
    # Plot the data
    bar_width = 0.35
    plt.bar(generation, worst, label='worst fittness', color='green')
    plt.bar(generation + bar_width, best, width=bar_width, label='best fittness', color='orange')
    plt.plot(generation, avg, label='average fittness', color='red')
    mode = ""
    if MODE == "C":
        mode = "CLASSIC"
    elif MODE == "L":
        mode = "LAMARK"
    elif MODE == "D":
        mode = "DARVIN"

    plt.title(f"{mode},POPULATION SIZE: {POPULATION_SIZE}, MUTATION RATE:{MUTATION_RATE}, ELITE SIZE:{ELITE_SIZE},"
              f"TOURNAMENT SIZE: {TOURNAMENT_SIZE}")
    plt.xlabel('Generation')
    plt.ylabel('Fittness score')

    # Set the x-limit of the first subplot
    plt.xlim(left=0)

    # Add a legend
    plt.legend()

    plt.figure()  # create a new figure
    steps = dict_graph[max_index][2]

    # Plot the data
    plt.bar(generation, steps, label='steps')

    plt.title(f"{mode},POPULATION SIZE: {POPULATION_SIZE}, MUTATION RATE:{MUTATION_RATE}, ELITE SIZE:{ELITE_SIZE},"
              f"TOURNAMENT SIZE: {TOURNAMENT_SIZE}")
    plt.xlabel('Generation')
    plt.ylabel('number of calls to fittness')

    # Add a legend
    plt.legend()
    plt.xlim(left=0)

    plt.show()


if __name__ == '__main__':
    """
       Main function that runs the genetic algorithm, performs additional operations, and creates a graph.

       Returns:
           None
    """

    run_number = 1
    if not gentic_algorithm(str(run_number)):
        while run_number < 5 and not STOP:
            run_number += 1
            MUTATION_RATE = random.uniform(MUTATION_RATE, 1)
            # steps = 0
            gentic_algorithm(str(run_number))

    max_pair = max(best_scores, key=lambda pair: pair[1])
    max_index = best_scores.index(max_pair) + 1  # get the index of the tuple with the maximum score

    best_permutation = dict(zip(max_pair[0], 'abcdefghijklmnopqrstuvwxyz'))

    # write the best permutation to file
    if best_permutation is not None:
        with open('perm.txt', 'w') as f:
            for k, v in sorted(best_permutation.items(), key=lambda item: item[0]):
                f.write(f'{k}: {v}\n')

    # output plaintext
    best_individual = max_pair[0]
    plaintext = ciphertext.translate(str.maketrans(best_individual, 'abcdefghijklmnopqrstuvwxyz'))
    with open('plaintext.txt', 'w') as f:
        f.write(plaintext)

    create_graph(str(max_index))
