from random import randint, sample, random

def run():
    generation = 0
    # Set up initial population
    population = Population(10)
    # Find current fittest
    most_fit = population.get_most_fit()
    # Loop
    while most_fit.fitness > 0:
        generation += 1
        population.selection()  # Could use alternate selections when close to converging
        if random() < 1.0:  # Probability of crossover
            offspring = population.crossover()
            if random() < 0.5:  # Probability of mutation
                offspring.mutation()
            population.find_not_fit()
            population.not_fit.chromosome = offspring.chromosome
        population.update_all_fitness()
        most_fit = population.get_most_fit()
        print("Generation: %d | Chromosome: %s | Fitness: %d" % (generation, most_fit.chromosome, most_fit.fitness))
    print("Converged on Generation: %d | Chromosome: %s" % (generation, most_fit.chromosome))
    return generation


def test_average_generations(number_trials):
    total_generations = 0
    for _ in range(0, number_trials):
        total_generations += run()
    print("Average Generations: %d" % (total_generations / number_trials))


def fitness(chromosome):
    score = ((chromosome[0] - 1) ** 2 + (chromosome[1] - 2) ** 2 + (chromosome[2] - 3) ** 2
             + (chromosome[3] - 4) ** 2 + (chromosome[4] - 5) ** 2 + (chromosome[5] - 10) ** 2)
    return score


class Population:

    def __init__(self, population_size):
        self.population = []
        for index in range(0, population_size):
            individual = Individual()
            print("%3d Chromosome: %-34s Fitness: %5d" % (index, individual.chromosome, individual.fitness))
            self.population.append(individual)
        self.fit_one = None
        self.fit_two = None
        self.not_fit = None

    def update_all_fitness(self):
        for individual in self.population:
            individual.fitness = fitness(individual.chromosome)

    def get_most_fit(self):
        return min(self.population, key=lambda individual: individual.fitness)

    def selection(self):
        # Tournament Selection
        tournament_one = sample(self.population, 3)
        tournament_two = sample(self.population, 3)
        # Find which is the most fit in each tournament
        self.fit_one = min(tournament_one, key=lambda individual: individual.fitness)
        self.fit_two = min(tournament_two, key=lambda individual: individual.fitness)
        # print("Parent ONE:   Chromosome: %-34s Fitness: %5d" % (self.fit_one.chromosome, self.fit_one.fitness))
        # print("Parent TWO:   Chromosome: %-34s Fitness: %5d" % (self.fit_two.chromosome, self.fit_two.fitness))

    def find_not_fit(self):
        # Tournament Selection
        tournament_not_fit = sample(self.population, 3)
        # Find which is the most fit in each tournament
        self.not_fit = max(tournament_not_fit, key=lambda individual: individual.fitness)

    def crossover(self):
        baby = Individual()
        for index in range(0, len(baby.chromosome)):
            baby.chromosome[index] = (self.fit_one.chromosome[index] + self.fit_two.chromosome[index]) // 2
        return baby


class Individual:

    def __init__(self):
        # Randomly initialize chromosome
        self.chromosome = []
        for _ in range(0, 6):
            self.chromosome.append(randint(-10, 20))
        # Find initial fitness
        self.fitness = fitness(self.chromosome)

    def mutation(self):
        mutate_index = randint(0, len(self.chromosome) - 1)
        self.chromosome[mutate_index] = randint(-10, 20)


if __name__ == '__main__':
    test_average_generations(20)
