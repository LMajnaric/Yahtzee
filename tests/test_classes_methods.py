# tests/test_classes_methods.py
import pytest
from unittest.mock import Mock, patch
from src.game_logic import GameLogic
from src.yahtzee_game import YahtzeeGame

class TestClassMethods:
    """Test class methods and interactions"""
    
    def test_game_logic_initialization(self):
        """Test GameLogic class initialization"""
        logic = GameLogic(num_dice=6)
        assert logic.num_dice == 6
        assert logic.max_rolls == 3
        assert not logic.state.game_over
        assert len(logic.state.scores) == 4  # 4 columns
    
    def test_method_chaining(self):
        """Test methods that depend on each other"""
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Test dice rolling -> scoring chain
        dice_result = game.roll_dice()
        assert len(dice_result.dice) == 5
        assert dice_result.roll_number == 1
        
        # Test that we can score after rolling
        available = game.get_available_categories(1)
        assert len(available) > 0
    
    @patch('random.randint')
    def test_dice_rolling_with_mock(self, mock_randint):
        """Test dice rolling with mocked random values"""
        # Mock dice to return specific values
        mock_randint.side_effect = [1, 2, 3, 4, 5]
        
        game = YahtzeeGame(num_dice=5)
        dice_result = game.roll_dice()
        
        assert dice_result.dice == [1, 2, 3, 4, 5]
        assert mock_randint.call_count == 5
    
    def test_error_handling(self):
        """Test error conditions and edge cases"""
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Test rolling too many times
        game.roll_dice()  # Roll 1
        game.roll_dice([0, 1])  # Roll 2
        game.roll_dice([0])  # Roll 3
        
        # Fourth roll should raise error
        with pytest.raises(ValueError, match="Maximum rolls exceeded"):
            game.roll_dice()

class TestPerformance:
    """Performance-related tests"""
    
    def test_large_number_of_games(self):
        """Test that we can run many games without issues"""
        import time
        
        start_time = time.time()
        total_score = 0
        
        for _ in range(100):  # Run 100 games
            game = YahtzeeGame(num_dice=5)
            game.start_new_game()
            
            # Quick game simulation
            for _ in range(48):  # Max possible turns
                if game.get_game_state().game_over:
                    break
                    
                game.roll_dice()
                
                # Score in first available category
                for col in range(1, 5):
                    available = game.get_available_categories(col)
                    if available:
                        if col == 4:
                            game.announce_category(col, available[0])
                        game.score_category(col, available[0])
                        break
            
            total_score += game.logic.get_total_score()
        
        end_time = time.time()
        avg_score = total_score / 100
        
        # Performance assertions
        assert end_time - start_time < 30  # Should complete in under 30 seconds
        assert avg_score > 50  # Average score should be reasonable
        
        print(f"100 games completed in {end_time - start_time:.2f} seconds")
        print(f"Average score: {avg_score:.2f}")