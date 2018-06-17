from random import randint, sample, random, uniform
from argparse import ArgumentParser
from sys import exit
from time import strftime, localtime


settings = {}
verbose = False

def setup_arguments():
    # Argument parser for command line arguments
    parser = ArgumentParser(description="Genetic Algorithm Tournament Script")
    parser.add_argument('-v', '--verbose', action='store_true', help="Display extra information when running")
    parser.add_argument('input', help="Input file")
    # parser.add_argument('output', help="Output file")
    args = parser.parse_args()
    print("Genetic Algorithm Tournament Script")
    print("Start: %s" % strftime("%d %b %Y %H:%M:%S %z\n", localtime()))
    # Set verbosity
    global verbose
    verbose = args.verbose
    # Read input file and assign to settings dictionary
    print("Reading input from %s" % args.input)
    try:
        with open(args.input) as file:
            for line in file:
                if not line.startswith('#') and line.strip():  # Ignore comments and blank lines
                    (key, val) = line.split()
                    settings[key] = val
    except IOError:
        exit("File not found")
    # except ValueError:
    #     exit("Input file formatted incorrectly")
    print("Settings: %s\n" % settings)
    # Check to see that all required settings were given to script
    required_settings = ['population_size', 'convergence_criterion', 'crossover_probability', 'mutation_probability',
                         'tournament_size', 'tournament_size_losers', 'gene_minvalue', 'gene_maxvalue',
                         'max_generations']
    for required_setting in required_settings:
        if required_setting not in settings:
            exit("Not all required settings found in input file")


def run():
    # Set up initial population
    generation = 0
    population = Population(int(settings['population_size']))
    # Find current fittest
    most_fit = population.get_most_fit()
    print("Generation: %d | Most Fit: %s | Fitness: %f\n" % (generation, most_fit.chromosome, most_fit.fitness))
    # Loop
    while most_fit.fitness > float(settings['convergence_criterion']) and generation < int(settings['max_generations']):
        generation += 1
        population.selection()  # Could use alternate selections when close to converging
        if random() < float(settings['crossover_probability']):  # Probability of crossover
            offspring = population.crossover()
            if random() < float(settings['mutation_probability']):  # Probability of mutation
                offspring.mutation()
            population.find_not_fit()
            population.not_fit.chromosome = offspring.chromosome
        population.update_all_fitness()
        most_fit = population.get_most_fit()
        print("Generation: %d | Most Fit: %s | Fitness: %f\n" % (generation, most_fit.chromosome, most_fit.fitness))
    if generation >= int(settings['max_generations']):
        print("Stop on Generation: %d | Chromosome: %s" % (generation, most_fit.chromosome))
    else:
        print("Converged on Generation: %d | Chromosome: %s" % (generation, most_fit.chromosome))
    print("End: %s" % strftime("%d %b %Y %H:%M:%S %z", localtime()))


def fitness(chromosome):
    # Define a custom fitness function here!
    score = ((chromosome[0] - 0.1) ** 2 + (chromosome[1] - 1.5) ** 2 + (chromosome[2] - 3.0) ** 2
             + (chromosome[3] - 4.44) ** 2 + (chromosome[4] - 5.0) ** 2 + (chromosome[5] - 10.0) ** 2)
    return score


class Population:

    def __init__(self, population_size):
        self.population = []
        if verbose:
            print("Creating population of size %d" % population_size)
        for index in range(0, population_size):
            individual = Individual()
            if verbose:
                print("Index: %d | Chromosome: %s | Fitness: %f" % (index, individual.chromosome, individual.fitness))
            self.population.append(individual)
        self.fit_one = None
        self.fit_two = None
        self.not_fit = None
        print('')

    def update_all_fitness(self):
        for individual in self.population:
            individual.fitness = fitness(individual.chromosome)

    def get_most_fit(self):
        return min(self.population, key=lambda individual: individual.fitness)

    def selection(self):
        # Tournament Selection
        tournament_one = sample(self.population, int(settings['tournament_size']))
        tournament_two = sample(self.population, int(settings['tournament_size']))
        # Find which is the most fit in each tournament
        self.fit_one = min(tournament_one, key=lambda individual: individual.fitness)
        self.fit_two = min(tournament_two, key=lambda individual: individual.fitness)
        if verbose:
            print("Parent ONE Chromosome: %s | Fitness: %f" % (self.fit_one.chromosome, self.fit_one.fitness))
            print("Parent TWO Chromosome: %s | Fitness: %f" % (self.fit_two.chromosome, self.fit_two.fitness))

    def find_not_fit(self):
        # Tournament Selection
        tournament_not_fit = sample(self.population, int(settings['tournament_size_losers']))
        # Find which is the most fit in each tournament
        self.not_fit = max(tournament_not_fit, key=lambda individual: individual.fitness)
        if verbose:
            print("Unfit chromosome to be replaced: %s | Fitness %f" % (self.not_fit.chromosome, self.not_fit.fitness))

    def crossover(self):
        # Make a new individual and set chromosome by averaging parents
        baby = Individual()
        for index in range(0, len(baby.chromosome)):
            baby.chromosome[index] = (self.fit_one.chromosome[index] + self.fit_two.chromosome[index]) / 2
        if verbose:
            print("New chromosome: %s" % baby.chromosome)
        return baby


class Individual:

    def __init__(self):
        # Randomly initialize chromosome
        self.chromosome = []
        for _ in range(0, 6):  # ATTENTION CHANGE GENE SETUP/NUMBER BASED ON FITNESS FUNCTION
            self.chromosome.append(uniform(float(settings['gene_minvalue']), float(settings['gene_maxvalue'])))
        # Find initial fitness
        self.fitness = fitness(self.chromosome)

    def mutation(self):
        mutate_index = randint(0, len(self.chromosome) - 1)
        self.chromosome[mutate_index] = uniform(float(settings['gene_minvalue']), float(settings['gene_maxvalue']))
        if verbose:
            print("Mutated chromosome: %s" % self.chromosome)


if __name__ == '__main__':
    setup_arguments()
    run()
