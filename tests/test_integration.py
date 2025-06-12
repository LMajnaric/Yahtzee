import pytest
from src.models import Category
from src.scoring import ScoreCalculator
from src.yahtzee_game import YahtzeeGame


class TestGameIntegration:
    """Integration tests for complete game scenarios"""
    
    def test_complete_game_flow(self):
        """Test a complete game from start to finish"""
        game = YahtzeeGame(num_dice=5)
        game.start_new_game()
        
        # Simulate filling all categories (simplified)
        all_categories = list(Category)
        
        for turn in range(48):  # 4 columns * 12 categories
            if game.get_game_state().game_over:
                break
                
            # Roll dice
            dice_result = game.roll_dice()
            
            # Find an available category in any column
            scored = False
            for col in range(1, 5):
                available = game.get_available_categories(col)
                if available:
                    category = available[0]
                    
                    # Handle column 4 announcement
                    if col == 4:
                        game.announce_category(col, category)
                    
                    success = game.score_category(col, category)
                    if success:
                        scored = True
                        break
            
            assert scored, f"Could not score on turn {turn}"
        
        # Game should be complete
        assert game.get_game_state().game_over
        
        # Total score should be reasonable (> 0)
        total_score = game.logic.get_total_score()
        assert total_score > 0

    def test_six_dice_vs_five_dice(self):
        """Test that 6-dice game behaves differently from 5-dice"""
        game_5 = YahtzeeGame(num_dice=5)
        game_6 = YahtzeeGame(num_dice=6)
        
        # Test max scoring with known dice
        test_dice_5 = [1, 2, 3, 4, 5]
        test_dice_6 = [1, 2, 3, 4, 5, 6]
        
        max_score_5 = ScoreCalculator.calculate_max_score(test_dice_5)
        max_score_6 = ScoreCalculator.calculate_max_score(test_dice_6)
        
        # 6-dice should give higher max score (best 5 of 6)
        assert max_score_6 > max_score_5
        assert max_score_5 == 15  # 1+2+3+4+5
        assert max_score_6 == 20  # 2+3+4+5+6 (excludes 1)

# Fixture for common test data
@pytest.fixture
def sample_game_state():
    """Create a sample game state for testing"""
    game = YahtzeeGame(num_dice=5)
    game.start_new_game()
    
    # Fill in some sample scores
    game.logic.state.scores[1][Category.ONES] = 3
    game.logic.state.scores[1][Category.TWOS] = 6
    game.logic.state.scores[1][Category.MAX] = 25
    game.logic.state.scores[1][Category.MIN] = 10
    
    game.logic.state.filled_categories[1].update([
        Category.ONES, Category.TWOS, Category.MAX, Category.MIN
    ])
    
    return game

class TestWithFixtures:
    """Tests using fixtures"""
    
    def test_fixture_usage(self, sample_game_state):
        """Test using the sample game state fixture"""
        game = sample_game_state
        state = game.get_game_state()
        
        assert state.scores[1][Category.ONES] == 3
        assert state.scores[1][Category.TWOS] == 6
        assert len(state.filled_categories[1]) == 4