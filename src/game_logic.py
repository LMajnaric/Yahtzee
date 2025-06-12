import random
from typing import List, Tuple, Optional, Dict, Set

from src.models import Category, DiceRoll, GameState
from src.scoring import ScoreCalculator



class ColumnConstraints:
    """Manages column-specific constraints and rules"""
    
    COLUMN_ORDER = {
        1: [Category.ONES, Category.TWOS, Category.THREES, Category.FOURS, 
            Category.FIVES, Category.SIXES, Category.MAX, Category.MIN,
            Category.STRAIGHT, Category.FULL_HOUSE, Category.POKER, Category.YAHTZEE],
        2: [],  # Any order
        3: [Category.YAHTZEE, Category.POKER, Category.FULL_HOUSE, Category.STRAIGHT,
            Category.MIN, Category.MAX, Category.SIXES, Category.FIVES,
            Category.FOURS, Category.THREES, Category.TWOS, Category.ONES],
        4: []   # Any order but requires announcement
    }
    
    @staticmethod
    def get_available_categories(column: int, filled_categories: Set[Category]) -> List[Category]:
        """Get available categories for a column"""
        all_categories = list(Category)
        available = [cat for cat in all_categories if cat not in filled_categories]
        
        if column in [2, 4]:  # Any order columns
            return available
        
        # For ordered columns, only return the next required category
        order = ColumnConstraints.COLUMN_ORDER[column]
        for category in order:
            if category not in filled_categories:
                return [category]
        return []
    
    @staticmethod
    def can_fill_category(column: int, category: Category, filled_categories: Set[Category]) -> bool:
        """Check if a category can be filled in a column"""
        if category in filled_categories:
            return False
        
        available = ColumnConstraints.get_available_categories(column, filled_categories)
        return category in available

class GameLogic:
    """Core game logic and state management"""
    
    def __init__(self, num_dice: int = 5):
        self.num_dice = num_dice
        self.max_rolls = 3
        self.reset_game()
    
    def reset_game(self):
        """Reset the game to initial state"""
        self.state = GameState(
            scores={i: {} for i in range(1, 5)},
            filled_categories={i: set() for i in range(1, 5)},
            current_turn=1,
            game_over=False,
            column_bonuses={i: 0 for i in range(1, 5)},
            special_scores={i: 0 for i in range(1, 5)}
        )
        self.current_dice = []
        self.current_roll = 0
        self.announcement_column = None
        self.announced_category = None
    
    def roll_dice(self, keep_indices: List[int] = None) -> DiceRoll:
        """Roll dice, optionally keeping some dice"""
        if self.current_roll >= self.max_rolls:
            raise ValueError("Maximum rolls exceeded")
        
        if keep_indices is None:
            keep_indices = []
        
        if self.current_roll == 0:
            # First roll - roll all dice
            self.current_dice = [random.randint(1, 6) for _ in range(self.num_dice)]
        else:
            # Subsequent rolls - keep specified dice, reroll others
            new_dice = self.current_dice.copy()
            for i in range(len(new_dice)):
                if i not in keep_indices:
                    new_dice[i] = random.randint(1, 6)
            self.current_dice = new_dice
        
        self.current_roll += 1
        return DiceRoll(self.current_dice.copy(), self.current_roll)
    
    def announce_category(self, column: int, category: Category) -> bool:
        """Announce category for column 4"""
        if column != 4:
            return False
        
        if not ColumnConstraints.can_fill_category(column, category, self.state.filled_categories[column]):
            return False
        
        self.announcement_column = column
        self.announced_category = category
        return True
    
    def score_category(self, column: int, category: Category) -> bool:
        """Score a category in a column"""
        if self.state.game_over:
            return False
        
        # Check column 4 announcement requirement
        if column == 4:
            if self.announcement_column != 4 or self.announced_category != category:
                return False
        
        if not ColumnConstraints.can_fill_category(column, category, self.state.filled_categories[column]):
            return False
        
        # Calculate score
        if column == 4 and self.announced_category == category:
            # For announced category, calculate actual score
            score = ScoreCalculator.calculate_score(self.current_dice, category, self.current_roll)
        elif column == 4:
            # For column 4 but wrong category, score is 0
            score = 0
        else:
            score = ScoreCalculator.calculate_score(self.current_dice, category, self.current_roll)
        
        # Record the score
        self.state.scores[column][category] = score
        self.state.filled_categories[column].add(category)
        
        # Check for bonuses
        self._check_column_bonus(column)
        
        # Reset for next turn
        self.current_dice = []
        self.current_roll = 0
        self.announcement_column = None
        self.announced_category = None
        self.state.current_turn += 1
        
        # Check if game is over
        if self._is_game_complete():
            self.state.game_over = True
            self._calculate_final_scores()
        
        return True
    
    def _check_column_bonus(self, column: int):
        """Check and apply column bonus for number categories"""
        number_categories = [Category.ONES, Category.TWOS, Category.THREES, 
                           Category.FOURS, Category.FIVES, Category.SIXES]
        
        number_total = sum(self.state.scores[column].get(cat, 0) 
                          for cat in number_categories 
                          if cat in self.state.scores[column])
        
        if number_total >= 60:
            self.state.column_bonuses[column] = 30
    
    def _calculate_final_scores(self):
        """Calculate special scores and final totals"""
        for column in range(1, 5):
            # Calculate special score: 1s count * (max - min)
            ones_score = self.state.scores[column].get(Category.ONES, 0)
            max_score = self.state.scores[column].get(Category.MAX, 0)
            min_score = self.state.scores[column].get(Category.MIN, 0)
            
            self.state.special_scores[column] = ones_score * (max_score - min_score)
    
    def _is_game_complete(self) -> bool:
        """Check if all categories are filled"""
        for column in range(1, 5):
            if len(self.state.filled_categories[column]) < 12:
                return False
        return True
    
    def get_column_total(self, column: int) -> int:
        """Get total score for a column"""
        base_score = sum(self.state.scores[column].values())
        bonus = self.state.column_bonuses[column]
        special = self.state.special_scores[column]
        return base_score + bonus + special
    
    def get_total_score(self) -> int:
        """Get total game score"""
        return sum(self.get_column_total(col) for col in range(1, 5))