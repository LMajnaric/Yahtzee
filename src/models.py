from typing import List, Optional, Dict, Set
from enum import Enum
from dataclasses import dataclass


class Category(Enum):
    ONES = "1s"
    TWOS = "2s" 
    THREES = "3s"
    FOURS = "4s"
    FIVES = "5s"
    SIXES = "6s"
    MAX = "max"
    MIN = "min"
    STRAIGHT = "straight"
    FULL_HOUSE = "full_house"
    POKER = "poker"
    YAHTZEE = "yahtzee"

class ColumnType(Enum):
    TOP_TO_BOTTOM = 1
    ANY_ORDER = 2
    BOTTOM_TO_TOP = 3
    ANNOUNCEMENT = 4
    
@dataclass
class DiceRoll:
    dice: List[int]
    roll_number: int  # 1, 2, or 3

@dataclass
class GameState:
    scores: Dict[int, Dict[Category, Optional[int]]]  # column -> category -> score
    filled_categories: Dict[int, Set[Category]]
    current_turn: int
    game_over: bool
    column_bonuses: Dict[int, int]
    special_scores: Dict[int, int]  # The 1s * (max - min) scoring