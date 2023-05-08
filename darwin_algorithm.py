import random

# read in the ciphertext, dictionary file, and letter frequency files
with open('enc.txt', 'r') as f:
    ciphertext = f.read().strip().lower()

with open('dict.txt', 'r') as f:
    dictionary = set(word.strip().lower() for word in f.readlines())

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

# define parameters for the genetic algorithm
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
ELITE_SIZE = 20
TOURNAMENT_SIZE = 10


# define fitness function
def fitness(plaintext):
    plaintext_freq = {}
    plaintext_pair_freq = {}
    plaintext_words = set(plaintext.split())
    num_words_in_dict = len(plaintext_words.intersection(dictionary))
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
        1/(1+abs(plaintext_freq.get(letter, 0) / count_alpha - letter_freq.get(letter, 0))) for letter in letter_freq)
    pair_fitness = sum(
        1/(1+abs(plaintext_pair_freq.get(pair, 0) / count_alpha - letter_pair_freq.get(pair, 0))) for pair in
        letter_pair_freq)
    return 2*num_words_in_dict + letter_fitness + pair_fitness


def new_word(word):
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
    cutoff = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cutoff] + parent2[cutoff:]
    child2 = parent2[:cutoff] + parent1[cutoff:]
    child1 = new_word(child1)
    child2 = new_word(child2)
    return child1, child2

# define mutation operator
def mutate(individual):
    mutated = list(individual)
    for i in range(len(mutated)):
        if random.random() < MUTATION_RATE:
            j = random.randint(0, len(mutated) - 1)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return ''.join(mutated)


def local_opt(individual, old_fittness):
    n = random.randint(5, 15)
    for i in range(n):
        mutated = list(individual)
        index1 = random.randint(0, len(mutated) - 1)
        index2 = index1
        while index1 == index2:
            index2 = random.randint(0, len(mutated) - 1)
        mutated[index1], mutated[index2] = mutated[index2], mutated[index1]
        new_individual = ''.join(mutated)
        new_fittness = fitness(ciphertext.translate(str.maketrans(new_individual, 'abcdefghijklmnopqrstuvwxyz')))
        if new_fittness > old_fittness:
            old_fittness = new_fittness
            individual = new_individual

    # new_individual = ''.join(mutated)
    # new_fittness = fitness(ciphertext.translate(str.maketrans(new_individual, 'abcdefghijklmnopqrstuvwxyz')))
    # if new_fittness > old_fittness:
    #     return new_individual, new_fittness
    return individual,  old_fittness


# generate initial population
population = [''.join(random.sample('abcdefghijklmnopqrstuvwxyz', 26)) for i in range(POPULATION_SIZE)]


best_permutation = None
best_fitness = float('-inf')
steps = 0
for generation in range(NUM_GENERATIONS):
    # calculate fitness for each individual
    steps += 1

    new_population = []
    fitnesses = []
    for individual in population:
        new_individual, new_fittness = local_opt(individual, fitness(
            ciphertext.translate(str.maketrans(individual, 'abcdefghijklmnopqrstuvwxyz'))))
        fitnesses.append((new_individual, new_fittness))

    fitnesses = [(individual, fitness(ciphertext.translate(str.maketrans(individual, 'abcdefghijklmnopqrstuvwxyz')))) for individual in population]

    fitnesses.sort(key=lambda x: x[1], reverse=True)
    if fitnesses[0][1] > best_fitness:
        best_individual, best_fitness = fitnesses[0]
        best_permutation = dict(zip(best_individual, 'abcdefghijklmnopqrstuvwxyz'))
    # select elite individuals
    elite = [individual for individual, fitness in fitnesses[:ELITE_SIZE]]
    # select parents via tournament selection
    parents = []
    for i in range(POPULATION_SIZE - ELITE_SIZE):
        tournament = random.sample(population, TOURNAMENT_SIZE)
        tournament_fitnesses = [(individual, fitness) for individual, fitness in fitnesses if individual in tournament]
        tournament_fitnesses.sort(key=lambda x: x[1], reverse=True)
        parents.append(tournament_fitnesses[0][0])
    # breed offspring via crossover
    offspring = []
    for i in range(0,POPULATION_SIZE - ELITE_SIZE, 2):
        parent1, parent2 = random.sample(parents, 2)
        child1, child2 = crossover(parent1, parent2)
        offspring.append(child1)
        offspring.append(child2)
    # apply mutation to offspring
    population = elite + [mutate(individual) for individual in offspring]
    print(f"Generation {generation} - Steps: {steps}, Best Fitness: {best_fitness}")
    print("p size is: ", len(population))

# write best permutation to file
if best_permutation is not None:
    with open('permutations.txt', 'w') as f:
        for k, v in sorted(best_permutation.items(), key=lambda item: item[1]):
            f.write(f'{v} {k}\n')
print(best_permutation)

# output plaintext
best_individual = fitnesses[0][0]
plaintext = ciphertext.translate(str.maketrans(best_individual, 'abcdefghijklmnopqrstuvwxyz'))
with open('plaintext.txt', 'w') as f:
    f.write(plaintext)