from typing import List, Tuple, Optional, Dict, Set



class DiceAnalyzer:
    """Analyzes dice combinations and calculates scores"""
    
    @staticmethod
    def get_best_five_dice(dice: List[int]) -> List[int]:
        """Select the best 5 dice from 6 for maximum scoring potential"""
        if len(dice) <= 5:
            return dice
        
        # For most categories, we want the highest values
        # But we need to consider what gives us the best scoring opportunities
        sorted_dice = sorted(dice, reverse=True)
        return sorted_dice[:5]
    
    @staticmethod
    def get_worst_five_dice(dice: List[int]) -> List[int]:
        """Select the worst 5 dice from 6 for minimum scoring"""
        if len(dice) <= 5:
            return dice
        
        sorted_dice = sorted(dice)
        return sorted_dice[:5]
    
    @staticmethod
    def count_occurrences(dice: List[int]) -> Dict[int, int]:
        """Count how many times each die value appears"""
        counts = {}
        for die in dice:
            counts[die] = counts.get(die, 0) + 1
        return counts
    
    @staticmethod
    def has_straight(dice: List[int]) -> bool:
        """Check if dice contain a 5-in-a-row straight"""
        unique_dice = set(dice)
        # Check for 1-2-3-4-5
        if {1, 2, 3, 4, 5}.issubset(unique_dice):
            return True
        # Check for 2-3-4-5-6
        if {2, 3, 4, 5, 6}.issubset(unique_dice):
            return True
        return False
    
    @staticmethod
    def has_full_house(dice: List[int]) -> Tuple[bool, List[int]]:
        """Check for full house and return the dice used"""
        counts = DiceAnalyzer.count_occurrences(dice)
        values = list(counts.values())
        
        if 3 in values and 2 in values:
            # Return the actual dice that form the full house
            three_kind = [k for k, v in counts.items() if v == 3][0]
            two_kind = [k for k, v in counts.items() if v == 2][0]
            result_dice = [three_kind] * 3 + [two_kind] * 2
            return True, result_dice
        return False, []
    
    @staticmethod
    def has_poker(dice: List[int]) -> Tuple[bool, int]:
        """Check for 4-of-a-kind and return the value"""
        counts = DiceAnalyzer.count_occurrences(dice)
        for value, count in counts.items():
            if count >= 4:
                return True, value
        return False, 0
    
    @staticmethod
    def has_yahtzee(dice: List[int]) -> Tuple[bool, int]:
        """Check for 5-of-a-kind and return the value"""
        counts = DiceAnalyzer.count_occurrences(dice)
        for value, count in counts.items():
            if count >= 5:
                return True, value
        return False, 0