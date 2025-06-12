import pytest
from src.dice_analysis import DiceAnalyzer


class TestDiceAnalysis:
    """Test dice analysis functions"""
    
    @pytest.mark.parametrize("dice,expected", [
        ([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2]),  # Best 5 from 6
        ([1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]),  # All same
        ([6, 6, 6, 6, 6], [6, 6, 6, 6, 6]),     # Already 5 dice
        ([2, 4, 6, 1, 3], [6, 4, 3, 2, 1]),     # 5 dice, should sort
    ])
    def test_get_best_five_dice(self, dice, expected):
        result = DiceAnalyzer.get_best_five_dice(dice)
        assert sorted(result, reverse=True) == sorted(expected, reverse=True)
    
    @pytest.mark.parametrize("dice,expected", [
        ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5]),  # Worst 5 from 6
        ([6, 6, 6, 6, 6, 6], [6, 6, 6, 6, 6]),  # All same
        ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]),     # Already 5 dice
    ])
    def test_get_worst_five_dice(self, dice, expected):
        result = DiceAnalyzer.get_worst_five_dice(dice)
        assert sorted(result) == sorted(expected)
    
    @pytest.mark.parametrize("dice,expected_straight", [
        ([1, 2, 3, 4, 5], True),      # 1-5 straight
        ([2, 3, 4, 5, 6], True),      # 2-6 straight
        ([1, 2, 3, 4, 5, 6], True),   # Both straights present
        ([1, 1, 2, 3, 4, 5], True),   # Straight with extra
        ([1, 2, 3, 4, 6], False),     # Missing 5
        ([1, 3, 4, 5, 6], False),     # Missing 2
        ([1, 1, 1, 1, 1], False),     # No straight
    ])
    def test_has_straight(self, dice, expected_straight):
        assert DiceAnalyzer.has_straight(dice) == expected_straight
    
    @pytest.mark.parametrize("dice,expected_has_fh,expected_dice", [
        ([2, 2, 2, 5, 5], True, [2, 2, 2, 5, 5]),
        ([6, 6, 6, 1, 1], True, [6, 6, 6, 1, 1]),
        ([1, 1, 2, 2, 3], False, []),
        ([5, 5, 5, 5, 5], False, []),  # Yahtzee, not full house
        ([1, 2, 3, 4, 5], False, []),  # Straight, not full house
    ])
    def test_has_full_house(self, dice, expected_has_fh, expected_dice):
        has_fh, fh_dice = DiceAnalyzer.has_full_house(dice)
        assert has_fh == expected_has_fh
        if expected_has_fh:
            assert sorted(fh_dice) == sorted(expected_dice)
    
    @pytest.mark.parametrize("dice,expected_has_poker,expected_value", [
        ([6, 6, 6, 6, 1], True, 6),   # 4 sixes
        ([2, 2, 2, 2, 2], True, 2),   # 5 twos (also poker)
        ([1, 1, 1, 5, 6], False, 0),  # Only 3 of a kind
        ([1, 2, 3, 4, 5], False, 0),  # No multiples
    ])
    def test_has_poker(self, dice, expected_has_poker, expected_value):
        has_poker, value = DiceAnalyzer.has_poker(dice)
        assert has_poker == expected_has_poker
        assert value == expected_value