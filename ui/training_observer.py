# ui/training_observer.py
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue

from src.models import Category

class TrainingObserver:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Yahtzee AI Training Observer")
        self.setup_ui()
        self.training_data = queue.Queue()
        
    def setup_ui(self):
        # Current game state display
        self.game_frame = ttk.LabelFrame(self.root, text="Current Game")
        self.game_frame.pack(padx=10, pady=5, fill="x")
        
        # Dice display
        self.dice_label = ttk.Label(self.game_frame, text="Dice: []", font=("Courier", 12))
        self.dice_label.pack()
        
        # Score display with 4 columns
        self.score_frame = ttk.Frame(self.game_frame)
        self.score_frame.pack(fill="both", expand=True)
        
        self.column_labels = []
        for i in range(4):
            col_frame = ttk.LabelFrame(self.score_frame, text=f"Column {i+1}")
            col_frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            label = ttk.Label(col_frame, text="", justify="left", font=("Courier", 8))
            label.pack()
            self.column_labels.append(label)
        
        # Training metrics
        self.metrics_frame = ttk.LabelFrame(self.root, text="Training Metrics")
        self.metrics_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        # Matplotlib figure for training progress
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, self.metrics_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Control buttons
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(padx=10, pady=5, fill="x")
        
        self.pause_btn = ttk.Button(self.control_frame, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=5)
        
        self.speed_scale = ttk.Scale(self.control_frame, from_=0.1, to=2.0, value=1.0, orient="horizontal")
        self.speed_scale.pack(side="left", padx=5, fill="x", expand=True)
        ttk.Label(self.control_frame, text="Speed").pack(side="left")
        
    def update_game_display(self, game_state, current_dice, ai_decision):
        """Update the current game display"""
        self.dice_label.config(text=f"Dice: {current_dice}")
        
        # Update each column's scores
        for col_idx in range(4):
            col_text = ""
            for category in Category:
                if category in game_state.scores[col_idx + 1]:
                    score = game_state.scores[col_idx + 1][category]
                    col_text += f"{category.value}: {score}\n"
                else:
                    col_text += f"{category.value}: -\n"
            
            col_text += f"\nBonus: {game_state.column_bonuses[col_idx + 1]}"
            col_text += f"\nSpecial: {game_state.special_scores[col_idx + 1]}"
            self.column_labels[col_idx].config(text=col_text)
    
    def update_training_metrics(self, episode, avg_score, learning_rate, epsilon=None):
        """Update training progress charts"""
        # TODO: This would plot training metrics over time
        pass
    
    def toggle_pause(self):
        # TODO: Implement pause/resume functionality
        pass

class TrainingVisualizer:
    """Manages the training loop with visualization"""
    
    def __init__(self, agent, observer):
        self.agent = agent
        self.observer = observer
        self.running = False
        self.paused = False
        
    def run_training_with_visualization(self, episodes=1000):
        """Run training while updating the observer"""
        self.running = True
        
        def training_loop():
            for episode in range(episodes):
                if not self.running:
                    break
                    
                while self.paused:
                    time.sleep(0.1)
                
                # Run one episode
                game_state, total_score = self.agent.play_episode()
                
                # Update observer
                self.observer.update_training_metrics(
                    episode, 
                    self.agent.get_average_score(),
                    self.agent.learning_rate
                )
                
                # Control speed
                speed = self.observer.speed_scale.get()
                time.sleep(1.0 / speed)
        
        # Run training in separate thread
        training_thread = threading.Thread(target=training_loop)
        training_thread.daemon = True
        training_thread.start()
        
        # Start UI main loop
        self.observer.root.mainloop()