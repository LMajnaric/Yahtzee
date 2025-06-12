class HybridYahtzeeAgent:
    def __init__(self):
        # Evolve the strategy parameters
        self.strategy_genome = {
            'early_game_aggression': 0.7,
            'column_priorities': [0.8, 0.6, 0.9, 0.4],  # Columns 1-4
            'risk_vs_safety_balance': 0.5,
            'combo_bonus_weight': 1.2
        }
        
        # Use neural network for tactical decisions
        self.tactical_network = SmallNN()
    
    def make_decision(self, game_state, dice):
        # High-level strategy from evolved genome
        strategy_context = self.get_strategy_context(game_state)
        
        # Tactical execution from neural network
        action = self.tactical_network.predict(game_state, dice, strategy_context)
        
        return action

class EvolutionaryTrainer:
    def evolve_strategies(self):
        # Evolve strategy genomes
        # Each genome gets a fresh neural network
        # Fitness = average performance over many games