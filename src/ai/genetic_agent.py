class YahtzeeGenome:
    def __init__(self):
        # Evolve decision rules, not just network weights
        self.dice_keep_threshold = random.random()  # When to keep dice
        self.category_priorities = [random.random() for _ in Category]
        self.risk_tolerance = random.random()
        
    def make_decision(self, game_state, dice):
        # Rule-based decisions using evolved parameters
        if self.evaluate_hand_strength(dice) > self.dice_keep_threshold:
            return "keep_all"
        # ... more rule logic
        
    def mutate(self):
        # Small random changes to parameters
        self.dice_keep_threshold += random.gauss(0, 0.1)
        # ... mutate other parameters

class GeneticTrainer:
    def evolve_population(self, population_size=100, generations=1000):
        population = [YahtzeeGenome() for _ in range(population_size)]
        
        for generation in range(generations):
            # Evaluate fitness (average game score)
            fitness_scores = []
            for genome in population:
                avg_score = self.evaluate_genome(genome)
                fitness_scores.append(avg_score)
            
            # Selection, crossover, mutation
            population = self.create_next_generation(population, fitness_scores)