from typing import List, Tuple, Optional, Dict, Set


from src.dice_analysis import DiceAnalyzer
from src.models import Category


class ScoreCalculator:
    """Calculates scores for different categories"""
    
    @staticmethod
    def calculate_number_score(dice: List[int], number: int) -> int:
        """Calculate score for number categories (1s, 2s, etc.)"""
        return sum(die for die in dice if die == number)
    
    @staticmethod
    def calculate_max_score(dice: List[int]) -> int:
        """Calculate max score (sum of best 5 dice)"""
        best_dice = DiceAnalyzer.get_best_five_dice(dice)
        return sum(best_dice)
    
    @staticmethod
    def calculate_min_score(dice: List[int]) -> int:
        """Calculate min score (sum of worst 5 dice)"""
        worst_dice = DiceAnalyzer.get_worst_five_dice(dice)
        return sum(worst_dice)
    
    @staticmethod
    def calculate_straight_score(dice: List[int], roll_number: int) -> int:
        """Calculate straight score based on roll number"""
        if not DiceAnalyzer.has_straight(dice):
            return 0
        
        if roll_number == 1:
            return 66
        elif roll_number == 2:
            return 55
        else:  # roll_number == 3
            return 46
    
    @staticmethod
    def calculate_full_house_score(dice: List[int]) -> int:
        """Calculate full house score"""
        has_fh, fh_dice = DiceAnalyzer.has_full_house(dice)
        if has_fh:
            return sum(fh_dice) + 30
        return 0
    
    @staticmethod
    def calculate_poker_score(dice: List[int]) -> int:
        """Calculate poker (4-of-a-kind) score"""
        has_poker, value = DiceAnalyzer.has_poker(dice)
        if has_poker:
            return 4 * value + 40
        return 0
    
    @staticmethod
    def calculate_yahtzee_score(dice: List[int]) -> int:
        """Calculate yahtzee (5-of-a-kind) score"""
        has_yahtzee, value = DiceAnalyzer.has_yahtzee(dice)
        if has_yahtzee:
            return 5 * value + 50
        return 0
    
    @staticmethod
    def calculate_score(dice: List[int], category: Category, roll_number: int) -> int:
        """Calculate score for any category"""
        if category == Category.ONES:
            return ScoreCalculator.calculate_number_score(dice, 1)
        elif category == Category.TWOS:
            return ScoreCalculator.calculate_number_score(dice, 2)
        elif category == Category.THREES:
            return ScoreCalculator.calculate_number_score(dice, 3)
        elif category == Category.FOURS:
            return ScoreCalculator.calculate_number_score(dice, 4)
        elif category == Category.FIVES:
            return ScoreCalculator.calculate_number_score(dice, 5)
        elif category == Category.SIXES:
            return ScoreCalculator.calculate_number_score(dice, 6)
        elif category == Category.MAX:
            return ScoreCalculator.calculate_max_score(dice)
        elif category == Category.MIN:
            return ScoreCalculator.calculate_min_score(dice)
        elif category == Category.STRAIGHT:
            return ScoreCalculator.calculate_straight_score(dice, roll_number)
        elif category == Category.FULL_HOUSE:
            return ScoreCalculator.calculate_full_house_score(dice)
        elif category == Category.POKER:
            return ScoreCalculator.calculate_poker_score(dice)
        elif category == Category.YAHTZEE:
            return ScoreCalculator.calculate_yahtzee_score(dice)