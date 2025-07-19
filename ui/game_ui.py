# ui/human_game_ui.py
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional
import random

from src.yahtzee_game import YahtzeeGame
from src.models import Category, GameState

class YahtzeeHumanUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Yahtzee Game")
        self.root.geometry("1000x700")
        
        self.game = YahtzeeGame(num_dice=6)  # Default to 6 dice
        self.current_dice = []
        self.current_roll = 0
        self.selected_dice = []
        self.announced_category = None
        
        self.setup_ui()
        self.new_game()
    
    def setup_ui(self):
        """Set up the complete UI"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Top section: Game controls and dice
        self.setup_game_controls(main_frame)
        self.setup_dice_area(main_frame)
        
        # Middle section: Score cards (4 columns)
        self.setup_score_cards(main_frame)
        
        # Bottom section: Game info and actions
        self.setup_action_area(main_frame)
    
    def setup_game_controls(self, parent):
        """Game controls section"""
        controls_frame = ttk.LabelFrame(parent, text="Game Controls")
        controls_frame.pack(fill="x", pady=(0, 10))
        
        # New game button
        ttk.Button(controls_frame, text="New Game", command=self.new_game).pack(side="left", padx=5)
        
        # Dice selection
        ttk.Label(controls_frame, text="Dice:").pack(side="left", padx=(20, 5))
        self.dice_var = tk.StringVar(value="6")
        dice_combo = ttk.Combobox(controls_frame, textvariable=self.dice_var, 
                                 values=["5", "6"], width=5, state="readonly")
        dice_combo.pack(side="left", padx=5)
        dice_combo.bind("<<ComboboxSelected>>", self.change_dice_count)
        
        # Game stats
        self.turn_label = ttk.Label(controls_frame, text="Turn: 1")
        self.turn_label.pack(side="right", padx=5)
        
        self.total_score_label = ttk.Label(controls_frame, text="Total Score: 0", 
                                          font=("Arial", 12, "bold"))
        self.total_score_label.pack(side="right", padx=20)
    
    def setup_dice_area(self, parent):
        """Dice rolling and selection area"""
        dice_frame = ttk.LabelFrame(parent, text="Dice")
        dice_frame.pack(fill="x", pady=(0, 10))
        
        # Roll info
        info_frame = ttk.Frame(dice_frame)
        info_frame.pack(fill="x", padx=5, pady=5)
        
        self.roll_label = ttk.Label(info_frame, text="Roll 0/3")
        self.roll_label.pack(side="left")
        
        # Roll button
        self.roll_button = ttk.Button(info_frame, text="Roll Dice", command=self.roll_dice)
        self.roll_button.pack(side="right")
        
        # Dice display
        self.dice_frame = ttk.Frame(dice_frame)
        self.dice_frame.pack(fill="x", padx=5, pady=5)
        
        self.dice_buttons = []
        self.dice_vars = []
    
    def setup_score_cards(self, parent):
        """Set up the 4 scoring columns"""
        scores_frame = ttk.LabelFrame(parent, text="Score Cards")
        scores_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Create 4 columns
        self.column_frames = []
        self.category_buttons = {}
        
        column_names = [
            "Column 1 (Top→Bottom)",
            "Column 2 (Any Order)", 
            "Column 3 (Bottom→Top)",
            "Column 4 (Announce)"
        ]
        
        for col in range(4):
            col_frame = ttk.LabelFrame(scores_frame, text=column_names[col])
            col_frame.grid(row=0, column=col, sticky="nsew", padx=5, pady=5)
            scores_frame.columnconfigure(col, weight=1)
            
            self.column_frames.append(col_frame)
            self.category_buttons[col + 1] = {}
            
            # Add category buttons for this column
            self.setup_column_categories(col_frame, col + 1)
        
        scores_frame.rowconfigure(0, weight=1)
    
    def setup_column_categories(self, parent, column):
        """Set up category buttons for a column"""
        categories = [
            ("1s", Category.ONES), ("2s", Category.TWOS), ("3s", Category.THREES),
            ("4s", Category.FOURS), ("5s", Category.FIVES), ("6s", Category.SIXES),
            ("Max", Category.MAX), ("Min", Category.MIN), 
            ("Straight", Category.STRAIGHT), ("Full House", Category.FULL_HOUSE),
            ("Poker", Category.POKER), ("Yahtzee", Category.YAHTZEE)
        ]
        
        for i, (name, category) in enumerate(categories):
            btn_frame = ttk.Frame(parent)
            btn_frame.pack(fill="x", padx=2, pady=1)
            
            # Category button
            btn = ttk.Button(btn_frame, text=name, width=12,
                           command=lambda c=column, cat=category: self.score_category(c, cat))
            btn.pack(side="left")
            
            # Score label
            score_label = ttk.Label(btn_frame, text="-", width=8, anchor="center", 
                                  relief="sunken", background="white")
            score_label.pack(side="right", padx=(5, 0))
            
            self.category_buttons[column][category] = {
                'button': btn,
                'score_label': score_label
            }
        
        # Column totals
        totals_frame = ttk.Frame(parent)
        totals_frame.pack(fill="x", padx=2, pady=(10, 2))
        
        ttk.Label(totals_frame, text="Bonus:", font=("Arial", 8, "bold")).pack(side="left")
        bonus_label = ttk.Label(totals_frame, text="0", width=8, anchor="center",
                              relief="sunken", background="lightblue")
        bonus_label.pack(side="right", padx=(5, 0))
        
        ttk.Label(totals_frame, text="Special:", font=("Arial", 8, "bold")).pack(side="left")
        special_label = ttk.Label(totals_frame, text="0", width=8, anchor="center",
                                relief="sunken", background="lightgreen")
        special_label.pack(side="right", padx=(5, 0))
        
        ttk.Label(totals_frame, text="Total:", font=("Arial", 10, "bold")).pack(side="left")
        total_label = ttk.Label(totals_frame, text="0", width=8, anchor="center",
                              relief="sunken", background="yellow", 
                              font=("Arial", 10, "bold"))
        total_label.pack(side="right", padx=(5, 0))
        
        self.category_buttons[column]['bonus_label'] = bonus_label
        self.category_buttons[column]['special_label'] = special_label
        self.category_buttons[column]['total_label'] = total_label
    
    def setup_action_area(self, parent):
        """Action area with announcements and possible scores"""
        action_frame = ttk.LabelFrame(parent, text="Actions & Information")
        action_frame.pack(fill="x")
        
        # Left side: Announcements for column 4
        announce_frame = ttk.Frame(action_frame)
        announce_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        ttk.Label(announce_frame, text="Column 4 Announcement:", 
                font=("Arial", 10, "bold")).pack(anchor="w")
        
        self.announce_var = tk.StringVar()
        self.announce_combo = ttk.Combobox(announce_frame, textvariable=self.announce_var,
                                        state="readonly", width=15)
        self.announce_combo.pack(anchor="w", pady=2)
        
        self.announce_button = ttk.Button(announce_frame, text="Announce", 
                                        command=self.make_announcement)
        self.announce_button.pack(anchor="w", pady=2)
        
        self.announcement_label = ttk.Label(announce_frame, text="No announcement", 
                                        foreground="red")
        self.announcement_label.pack(anchor="w", pady=2)
        
        # Right side: Possible scores
        scores_info_frame = ttk.Frame(action_frame)
        scores_info_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        ttk.Label(scores_info_frame, text="Possible Scores:", 
                font=("Arial", 10, "bold")).pack(anchor="w")
        
        self.possible_scores_text = tk.Text(scores_info_frame, height=4, width=40)
        self.possible_scores_text.pack(fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(scores_info_frame, orient="vertical", 
                                command=self.possible_scores_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.possible_scores_text.configure(yscrollcommand=scrollbar.set)
    
    def new_game(self):
        """Start a new game"""
        num_dice = int(self.dice_var.get())
        self.game = YahtzeeGame(num_dice=num_dice)
        self.game.start_new_game()
        
        self.current_dice = []
        self.current_roll = 0
        self.selected_dice = []
        self.announced_category = None
        
        # Reset announcement controls
        self.announce_button.config(state="disabled")
        self.announce_combo.config(state="disabled")
        self.announcement_label.config(text="No announcement", foreground="red")
        
        self.update_display()
        self.create_dice_buttons()
        
    def change_dice_count(self, event=None):
        """Handle dice count change"""
        if messagebox.askyesno("Change Dice Count", 
                              "Starting a new game will reset your progress. Continue?"):
            self.new_game()
    
    def create_dice_buttons(self):
        """Create dice selection buttons"""
        # Clear existing dice buttons
        for widget in self.dice_frame.winfo_children():
            widget.destroy()
        
        self.dice_buttons = []
        self.dice_vars = []
        
        num_dice = int(self.dice_var.get())
        for i in range(num_dice):
            var = tk.BooleanVar()
            self.dice_vars.append(var)
            
            btn = tk.Checkbutton(self.dice_frame, text="?", font=("Arial", 16, "bold"),
                               width=4, height=2, variable=var,
                               command=self.update_selected_dice)
            btn.pack(side="left", padx=2)
            self.dice_buttons.append(btn)
    
    def roll_dice(self):
        """Roll the dice"""
        if self.current_roll >= 3:
            messagebox.showwarning("Max Rolls", "You've already rolled 3 times!")
            return
        
        try:
            if self.current_roll == 0:
                # First roll
                dice_result = self.game.roll_dice()
            else:
                # Subsequent rolls - keep selected dice
                keep_indices = [i for i, var in enumerate(self.dice_vars) if var.get()]
                dice_result = self.game.roll_dice(keep_indices)
            
            self.current_dice = dice_result.dice
            self.current_roll = dice_result.roll_number
            
            self.update_dice_display()
            self.update_possible_scores()
            self.update_roll_info()
            self.update_announce_options()
            
            # Enable announcements only after first roll, disable after first roll
            if self.current_roll == 1:
                self.announce_button.config(state="normal")
                self.announce_combo.config(state="readonly")
            else:
                # Disable announcements after first roll if not already announced
                if not self.announced_category:
                    self.announce_button.config(state="disabled")
                    self.announce_combo.config(state="disabled")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error rolling dice: {e}")
    
    def update_dice_display(self):
        """Update the dice display"""
        for i, (btn, die_value) in enumerate(zip(self.dice_buttons, self.current_dice)):
            btn.config(text=str(die_value))
            # Reset selection for new roll
            if self.current_roll == 1:
                self.dice_vars[i].set(False)
    
    def update_selected_dice(self):
        """Update which dice are selected"""
        self.selected_dice = [i for i, var in enumerate(self.dice_vars) if var.get()]
    
    def update_possible_scores(self):
        """Update the possible scores display"""
        if not self.current_dice:
            return
        
        possible_scores = self.game.get_possible_scores(self.current_dice, self.current_roll)
        
        self.possible_scores_text.delete(1.0, tk.END)
        for category, score in possible_scores.items():
            self.possible_scores_text.insert(tk.END, f"{category.value}: {score}\n")
    
    def update_announce_options(self):
        """Update announcement options for column 4"""
        available = self.game.get_available_categories(4)
        category_names = [cat.value for cat in available]
        
        self.announce_combo.config(values=category_names)
        if category_names:
            self.announce_combo.set(category_names[0])
    
    def make_announcement(self):
        """Make an announcement for column 4"""
        # Can only announce after first roll
        if self.current_roll != 1:
            messagebox.showwarning("Invalid Timing", 
                                "You can only announce after the first roll!")
            return
        
        if not self.announce_var.get():
            messagebox.showwarning("No Selection", "Please select a category to announce!")
            return
        
        # Find the category enum from the string
        category_name = self.announce_var.get()
        category = None
        for cat in Category:
            if cat.value == category_name:
                category = cat
                break
        
        if category and self.game.announce_category(4, category):
            self.announced_category = category
            self.announcement_label.config(text=f"Announced: {category_name}", 
                                        foreground="green")
            
            # Disable announcement controls after announcing
            self.announce_button.config(state="disabled")
            self.announce_combo.config(state="disabled")
            
            # Immediately update button states to enable the announced category
            self.update_button_states()
            
            messagebox.showinfo("Announcement", f"You announced: {category_name}")
        else:
            messagebox.showerror("Invalid", "Cannot announce this category!")
    
    def score_category(self, column, category):
        """Score a category"""
        if not self.current_dice:
            messagebox.showwarning("No Dice", "Please roll dice first!")
            return
        
        # Check if column 4 needs announcement
        if column == 4 and not self.announced_category:
            messagebox.showwarning("No Announcement", 
                                 "You must announce a category for Column 4 first!")
            return
        
        # Check if this is the announced category for column 4
        if column == 4 and category != self.announced_category:
            messagebox.showwarning("Wrong Category", 
                                 f"You announced {self.announced_category.value}!")
            return
        
        success = self.game.score_category(column, category)
        
        if success:
            self.update_display()
            self.reset_turn()
            
            if self.game.get_game_state().game_over:
                self.game_over()
        else:
            messagebox.showerror("Invalid Move", "Cannot score this category!")
    
    def reset_turn(self):
        """Reset for next turn"""
        self.current_dice = []
        self.current_roll = 0
        self.selected_dice = []
        self.announced_category = None
        
        # Clear dice display
        for btn in self.dice_buttons:
            btn.config(text="?")
        for var in self.dice_vars:
            var.set(False)
        
        # Reset announcements
        self.announcement_label.config(text="No announcement", foreground="red")
        self.announce_button.config(state="disabled")
        self.announce_combo.config(state="disabled")
        self.possible_scores_text.delete(1.0, tk.END)
        
        self.update_roll_info()
        self.update_button_states()

    def update_display(self):
        """Update all display elements"""
        state = self.game.get_game_state()
        
        # Update turn and total score
        self.turn_label.config(text=f"Turn: {state.current_turn}")
        total_score = self.game.logic.get_total_score()
        self.total_score_label.config(text=f"Total Score: {total_score}")
        
        # Update score cards
        for column in range(1, 5):
            self.update_column_display(column, state)
        
        # Update button states
        self.update_button_states()
    
    def update_column_display(self, column, state):
        """Update display for one column"""
        # Update category scores
        for category in Category:
            if category in state.scores[column]:
                score = state.scores[column][category]
                self.category_buttons[column][category]['score_label'].config(text=str(score))
            
            # Update button state (enabled/disabled) - but NOT for column 4 here
            # Column 4 is handled separately in update_button_states()
            if column != 4:
                available = self.game.get_available_categories(column)
                if category in available:
                    self.category_buttons[column][category]['button'].config(state="normal")
                else:
                    self.category_buttons[column][category]['button'].config(state="disabled")
        
        # Update totals
        bonus = state.column_bonuses[column]
        special = state.special_scores[column]
        total = self.game.logic.get_column_total(column)
        
        self.category_buttons[column]['bonus_label'].config(text=str(bonus))
        self.category_buttons[column]['special_label'].config(text=str(special))
        self.category_buttons[column]['total_label'].config(text=str(total))
    
    def update_button_states(self):
        """Update button enabled/disabled states"""
        # Enable/disable roll button
        if self.current_roll >= 3:
            self.roll_button.config(text="Must Score", state="disabled")
        else:
            self.roll_button.config(text="Roll Dice", state="normal")
        
        # If column 4 announcement is made, disable all other columns
        if self.announced_category:
            # Disable all buttons in columns 1, 2, 3
            for column in range(1, 4):
                for category in Category:
                    self.category_buttons[column][category]['button'].config(state="disabled")
            
            # Handle column 4: only enable the announced category
            available_col4 = self.game.get_available_categories(4)
            for category in Category:
                if category == self.announced_category and category in available_col4:
                    self.category_buttons[4][category]['button'].config(state="normal")
                else:
                    self.category_buttons[4][category]['button'].config(state="disabled")
        
        else:
            # No announcement made, handle all columns normally
            
            # Handle columns 1, 2, 3 normally
            for column in range(1, 4):
                available = self.game.get_available_categories(column)
                for category in Category:
                    if category in available:
                        self.category_buttons[column][category]['button'].config(state="normal")
                    else:
                        self.category_buttons[column][category]['button'].config(state="disabled")
            
            # Handle column 4: disable all buttons when no announcement
            for category in Category:
                self.category_buttons[4][category]['button'].config(state="disabled")
    
    def update_roll_info(self):
        """Update roll information display"""
        self.roll_label.config(text=f"Roll {self.current_roll}/3")
        
        if self.current_roll >= 3:
            self.roll_button.config(state="disabled", text="Must Score Now!")
        else:
            self.roll_button.config(state="normal", text="Roll Dice")
    
    def game_over(self):
        """Handle game over"""
        final_score = self.game.logic.get_total_score()
        messagebox.showinfo("Game Over!", 
                          f"Game Complete!\nFinal Score: {final_score}")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

# Main entry point
if __name__ == "__main__":
    app = YahtzeeHumanUI()
    app.run()