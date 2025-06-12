from typing import List, Tuple, Optional, Dict, Set


from src.game_logic import ColumnConstraints, GameLogic
from src.models import Category, DiceRoll, GameState
from src.scoring import ScoreCalculator


class YahtzeeGame:
    """Main game interface"""
    
    def __init__(self, num_dice: int = 5):
        self.logic = GameLogic(num_dice)
    
    def start_new_game(self):
        """Start a new game"""
        self.logic.reset_game()
    
    def get_game_state(self) -> GameState:
        """Get current game state"""
        return self.logic.state
    
    def get_available_categories(self, column: int) -> List[Category]:
        """Get available categories for a column"""
        return ColumnConstraints.get_available_categories(
            column, self.logic.state.filled_categories[column]
        )
    
    def roll_dice(self, keep_indices: List[int] = None) -> DiceRoll:
        """Roll dice"""
        return self.logic.roll_dice(keep_indices)
    
    def announce_category(self, column: int, category: Category) -> bool:
        """Announce category for column 4"""
        return self.logic.announce_category(column, category)
    
    def score_category(self, column: int, category: Category) -> bool:
        """Score a category"""
        return self.logic.score_category(column, category)
    
    def get_possible_scores(self, dice: List[int], roll_number: int) -> Dict[Category, int]:
        """Get possible scores for current dice (useful for AI)"""
        scores = {}
        for category in Category:
            scores[category] = ScoreCalculator.calculate_score(dice, category, roll_number)
        return scores