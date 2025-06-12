# tests/test_scoring.py
import pytest
from src.scoring import ScoreCalculator
from src.dice_analysis import DiceAnalyzer
from src.models import Category
from src.yahtzee_game import YahtzeeGame

class TestScoring:
    """Test scoring calculations"""
    
    @pytest.mark.parametrize("dice,number,expected_score", [
        ([1, 1, 2, 3, 4], 1, 2),      # Two 1s
        ([6, 6, 6, 6, 6], 6, 30),     # Five 6s
        ([1, 2, 3, 4, 5], 6, 0),      # No 6s
        ([3, 3, 3, 1, 2], 3, 9),      # Three 3s
    ])
    def test_number_scoring(self, dice, number, expected_score):
        assert ScoreCalculator.calculate_number_score(dice, number) == expected_score
    
    @pytest.mark.parametrize("dice,expected_score", [
        ([1, 2, 3, 4, 5], 15),        # Sum of all
        ([6, 6, 6, 6, 6], 30),        # All 6s
        ([1, 2, 3, 4, 5, 6], 20),     # Best 5 from 6 (excludes 1)
        ([1, 1, 1, 6, 6, 6], 20),     # Best 5 from 6 (three 6s, two 1s)
    ])
    def test_max_scoring(self, dice, expected_score):
        assert ScoreCalculator.calculate_max_score(dice) == expected_score
    
    @pytest.mark.parametrize("dice,expected_score", [
        ([1, 2, 3, 4, 5], 15),        # Sum of all (same as max for 5 dice)
        ([6, 6, 6, 6, 6], 30),        # All 6s
        ([1, 2, 3, 4, 5, 6], 15),     # Worst 5 from 6 (excludes 6)
        ([1, 1, 1, 6, 6, 6], 15),      # Worst 5 from 6 (three 1s, two 6s)
    ])
    def test_min_scoring(self, dice, expected_score):
        assert ScoreCalculator.calculate_min_score(dice) == expected_score
    
    @pytest.mark.parametrize("dice,roll_number,expected_score", [
        ([1, 2, 3, 4, 5], 1, 66),     # First roll straight
        ([1, 2, 3, 4, 5], 2, 55),     # Second roll straight
        ([1, 2, 3, 4, 5], 3, 46),     # Third roll straight
        ([2, 3, 4, 5, 6], 1, 66),     # Different straight, first roll
        ([1, 2, 3, 4, 6], 1, 0),      # No straight
    ])
    def test_straight_scoring(self, dice, roll_number, expected_score):
        assert ScoreCalculator.calculate_straight_score(dice, roll_number) == expected_score
    
    def test_full_house_scoring_example(self):
        """Test the specific example from requirements: [5,5,6,6,6] = 28+30=58"""
        dice = [5, 5, 6, 6, 6]
        score = ScoreCalculator.calculate_full_house_score(dice)
        assert score == 58  # sum(dice) + 30 = 27 + 30
    
    @pytest.mark.parametrize("dice,expected_score", [
        ([5, 5, 6, 6, 6], 58),        # Example from requirements
        ([1, 1, 1, 2, 2], 37),        # 1+1+1+2+2 = 7, +30 = 37
        ([2, 2, 3, 3, 3], 43),        # 2+2+3+3+3 = 13, +30 = 43
        ([1, 2, 3, 4, 5], 0),         # No full house
        ([6, 6, 6, 6, 6], 0),         # Yahtzee, not full house
    ])
    def test_full_house_scoring_parametrized(self, dice, expected_score):
        score = ScoreCalculator.calculate_full_house_score(dice)
        assert score == expected_score
    
    @pytest.mark.parametrize("dice,expected_score", [
        ([6, 6, 6, 6, 1], 64),        # 4*6 + 40 = 64
        ([3, 3, 3, 3, 5], 52),        # 4*3 + 40 = 52
        ([1, 1, 1, 1, 1], 44),        # 4*1 + 40 = 44 (5 of a kind counts as poker too)
        ([1, 2, 3, 4, 5], 0),         # No poker
    ])
    def test_poker_scoring(self, dice, expected_score):
        assert ScoreCalculator.calculate_poker_score(dice) == expected_score
    
    @pytest.mark.parametrize("dice,expected_score", [
        ([6, 6, 6, 6, 6], 80),        # 5*6 + 50 = 80
        ([1, 1, 1, 1, 1], 55),        # 5*1 + 50 = 55
        ([3, 3, 3, 3, 3], 65),        # 5*3 + 50 = 65
        ([1, 1, 1, 1, 2], 0),         # Only 4 of a kind
    ])
    def test_yahtzee_scoring(self, dice, expected_score):
        assert ScoreCalculator.calculate_yahtzee_score(dice) == expected_score

class TestSpecialColumnScoring:
    """Test the special column scoring rule: 1s * (max - min)"""
    
    def test_special_column_scoring_basic(self):
        """Test basic special scoring calculation"""
        # Create a game state with some scores
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Manually set some scores for column 1
        game.logic.state.scores[1][Category.ONES] = 3  # Got 3 ones
        game.logic.state.scores[1][Category.MAX] = 25
        game.logic.state.scores[1][Category.MIN] = 10
        
        # Calculate special score
        game.logic._calculate_final_scores()
        
        # Should be 3 * (25 - 10) = 45
        assert game.logic.state.special_scores[1] == 45
    
    @pytest.mark.parametrize("ones_score,max_score,min_score,expected_special", [
        (0, 30, 15, 0),     # No ones = no special score
        (5, 30, 15, 75),    # 5 * (30-15) = 75
        (2, 20, 20, 0),     # Same max/min = 0
        (4, 25, 10, 60),    # 4 * (25-10) = 60
        (1, 30, 5, 25),     # 1 * (30-5) = 25
    ])
    def test_special_scoring_parametrized(self, ones_score, max_score, min_score, expected_special):
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Set scores for column 1
        game.logic.state.scores[1][Category.ONES] = ones_score
        game.logic.state.scores[1][Category.MAX] = max_score
        game.logic.state.scores[1][Category.MIN] = min_score
        
        game.logic._calculate_final_scores()
        
        assert game.logic.state.special_scores[1] == expected_special
    
    def test_special_scoring_all_columns(self):
        """Test special scoring across all columns"""
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Set different scores for each column
        test_data = [
            (2, 30, 10),  # Column 1: 2 * (30-10) = 40
            (0, 25, 15),  # Column 2: 0 * (25-15) = 0
            (4, 20, 5),   # Column 3: 4 * (20-5) = 60
            (3, 28, 12),  # Column 4: 3 * (28-12) = 48
        ]
        
        expected_specials = [40, 0, 60, 48]
        
        for col, (ones, max_val, min_val) in enumerate(test_data, 1):
            game.logic.state.scores[col][Category.ONES] = ones
            game.logic.state.scores[col][Category.MAX] = max_val
            game.logic.state.scores[col][Category.MIN] = min_val
        
        game.logic._calculate_final_scores()
        
        for col, expected in enumerate(expected_specials, 1):
            assert game.logic.state.special_scores[col] == expected

class TestColumnConstraints:
    """Test column-specific rules and constraints"""
    
    def test_column_1_ordering(self):
        """Test that column 1 must be filled top to bottom"""
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Should only be able to fill ONES first
        available = game.get_available_categories(1)
        assert available == [Category.ONES]
        
        # After filling ONES, should only be able to fill TWOS
        game.logic.state.filled_categories[1].add(Category.ONES)
        available = game.get_available_categories(1)
        assert available == [Category.TWOS]
    
    def test_column_2_any_order(self):
        """Test that column 2 can be filled in any order"""
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Should be able to fill any category
        available = game.get_available_categories(2)
        assert len(available) == 12  # All categories available
        
        # After filling one category, others should still be available
        game.logic.state.filled_categories[2].add(Category.YAHTZEE)
        available = game.get_available_categories(2)
        assert len(available) == 11  # All except YAHTZEE
        assert Category.YAHTZEE not in available
    
    def test_column_3_reverse_ordering(self):
        """Test that column 3 must be filled bottom to top"""
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Should only be able to fill YAHTZEE first
        available = game.get_available_categories(3)
        assert available == [Category.YAHTZEE]
        
        # After filling YAHTZEE, should only be able to fill POKER
        game.logic.state.filled_categories[3].add(Category.YAHTZEE)
        available = game.get_available_categories(3)
        assert available == [Category.POKER]
    
    def test_column_4_announcement_required(self):
        """Test that column 4 requires announcement"""
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Roll some dice
        game.roll_dice()
        
        # Try to score without announcement - should fail
        success = game.score_category(4, Category.ONES)
        assert not success
        
        # Announce category then score - should work
        game.announce_category(4, Category.ONES)
        success = game.score_category(4, Category.ONES)
        assert success

class TestBonusCalculation:
    """Test bonus calculations"""
    
    @pytest.mark.parametrize("number_scores,expected_bonus", [
        ([5, 10, 15, 20, 10, 5], 30),    # 65 total >= 60, gets bonus
        ([3, 6, 9, 12, 15, 18], 30),     # 63 total >= 60, gets bonus
        ([1, 2, 3, 4, 5, 6], 0),         # 21 total < 60, no bonus
        ])
    def test_column_bonus_calculation(self, number_scores, expected_bonus):
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Set number category scores
        number_categories = [Category.ONES, Category.TWOS, Category.THREES, 
                           Category.FOURS, Category.FIVES, Category.SIXES]
        
        for i, score in enumerate(number_scores):
            game.logic.state.scores[1][number_categories[i]] = score
        
        # Check bonus calculation
        game.logic._check_column_bonus(1)
        assert game.logic.state.column_bonuses[1] == expected_bonus