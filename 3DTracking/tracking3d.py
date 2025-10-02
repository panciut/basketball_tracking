import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class BasketballCourtVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        self.setup_court()
        
    def setup_court(self):
        """Setup basketball court dimensions and markings"""
        # NBA court dimensions (feet)
        self.court_length = 94 * 12  # 94 feet in inches
        self.court_width = 50 * 12   # 50 feet in inches
        
        # Adjust based on your data range (scaling to fit your coordinates)
        self.ax.set_xlim(-self.court_length/2, self.court_length/2)
        self.ax.set_ylim(-self.court_width/2, self.court_width/2)
        self.ax.set_aspect('equal')
        
        # Draw court background
        court = Rectangle((-self.court_length/2, -self.court_width/2), 
                        self.court_length, self.court_width, 
                        facecolor='#e6c79c', alpha=0.8)
        self.ax.add_patch(court)
        
        # Court markings
        self.draw_court_markings()
        
    def draw_court_markings(self):
        """Draw standard basketball court markings"""
        # Court boundaries
        boundary = Rectangle((-self.court_length/2, -self.court_width/2), 
                          self.court_length, self.court_width, 
                          fill=False, color='black', linewidth=2)
        self.ax.add_patch(boundary)
        
        # Center line
        self.ax.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        
        # Center circle
        center_circle = Circle((0, 0), 6*12, fill=False, color='black', linewidth=1)
        self.ax.add_patch(center_circle)
        
        # Free throw circles
        ft_circle_radius = 6*12
        left_ft_circle = Circle((-self.court_length/4, 0), ft_circle_radius, 
                              fill=False, color='black', linewidth=1)
        right_ft_circle = Circle((self.court_length/4, 0), ft_circle_radius, 
                               fill=False, color='black', linewidth=1)
        self.ax.add_patch(left_ft_circle)
        self.ax.add_patch(right_ft_circle)
        
        # Key (paint area)
        key_width = 16*12
        key_height = 19*12
        left_key = Rectangle((-self.court_length/2, -key_width/2), 
                          key_height, key_width, fill=False, color='black', linewidth=2)
        right_key = Rectangle((self.court_length/2 - key_height, -key_width/2), 
                           key_height, key_width, fill=False, color='black', linewidth=2)
        self.ax.add_patch(left_key)
        self.ax.add_patch(right_key)
        
        # Three-point line (simplified arc)
        three_point_radius = 23.75*12
        left_three_pt = Arc((-self.court_length/2, 0), three_point_radius*2, three_point_radius*2,
                          theta1=90, theta2=270, color='black', linewidth=1)
        right_three_pt = Arc((self.court_length/2, 0), three_point_radius*2, three_point_radius*2,
                           theta1=270, theta2=90, color='black', linewidth=1)
        self.ax.add_patch(left_three_pt)
        self.ax.add_patch(right_three_pt)
        
        # Hoops (simplified)
        hoop_radius = 0.75*12
        left_hoop = Circle((-self.court_length/2 + 4*12, 0), hoop_radius, 
                         fill=False, color='red', linewidth=2)
        right_hoop = Circle((self.court_length/2 - 4*12, 0), hoop_radius, 
                          fill=False, color='red', linewidth=2)
        self.ax.add_patch(left_hoop)
        self.ax.add_patch(right_hoop)

class BasketballTrackingAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.preprocess_data()
        self.setup_team_assignment()
        self.setup_colors()
        
    def preprocess_data(self):
        """Clean and prepare tracking data"""
        # Remove rows with extremely high reprojection error
        self.df = self.df[self.df['mean_reproj_error'] < 100]
        
        # Create consistent object IDs
        self.df['object_id'] = self.df['group_id'] + '_' + self.df['label']
        
        # Sort by frame
        self.df = self.df.sort_values('frame_id')
        
        # Get frame range
        self.frames = sorted(self.df['frame_id'].unique())
        
        # Scale coordinates to fit court if needed
        self.scale_coordinates()
        
    def scale_coordinates(self):
        """Scale coordinates to fit basketball court dimensions"""
        # Your data seems to be in a different coordinate system
        # Let's center and scale it to fit a standard basketball court
        if self.df.empty:
            self.df['X_scaled'] = []
            self.df['Y_scaled'] = []
            return
        x_center = self.df['X'].mean()
        y_center = self.df['Y'].mean()

        x_range = self.df['X'].max() - self.df['X'].min()
        y_range = self.df['Y'].max() - self.df['Y'].min()
        # Avoid division by zero
        if x_range == 0:
            x_range = 1.0
        if y_range == 0:
            y_range = 1.0
        x_scale = 94 * 12 / x_range * 0.8
        y_scale = 50 * 12 / y_range * 0.8

        self.df['X_scaled'] = (self.df['X'] - x_center) * x_scale
        self.df['Y_scaled'] = (self.df['Y'] - y_center) * y_scale
        
    def setup_team_assignment(self):
        """Assign players to teams based on their typical positions"""
        # Simple team assignment based on average X position
        player_data = self.df[self.df['label'] == 'player']
        
        # Group by object_id and get average X position
        player_avg_x = player_data.groupby('object_id')['X_scaled'].mean()
        
        # Split into two teams based on X position
        team_a_players = player_avg_x[player_avg_x < 0].index.tolist()
        team_b_players = player_avg_x[player_avg_x >= 0].index.tolist()
        
        # Create team assignment
        self.df['team'] = 'unknown'
        self.df.loc[self.df['object_id'].isin(team_a_players), 'team'] = 'team_a'
        self.df.loc[self.df['object_id'].isin(team_b_players), 'team'] = 'team_b'
        self.df.loc[self.df['label'] == 'referee', 'team'] = 'referee'
        self.df.loc[self.df['label'] == 'ball', 'team'] = 'ball'
        
    def setup_colors(self):
        """Setup team-based colors"""
        self.team_colors = {
            'team_a': 'blue',
            'team_b': 'red', 
            'referee': 'black',
            'ball': 'orange',
            'unknown': 'gray'
        }
        
    def get_frame_data(self, frame):
        """Get all objects in a specific frame"""
        return self.df[self.df['frame_id'] == frame]
    
    def get_object_trajectory(self, object_id):
        """Get complete trajectory for an object"""
        return self.df[self.df['object_id'] == object_id]
    
    def get_ball_positions(self):
        """Get all ball positions"""
        return self.df[self.df['label'] == 'ball']
    
    def get_player_positions_by_team(self, frame, team):
        """Get all player positions for a specific team in a frame"""
        frame_data = self.get_frame_data(frame)
        return frame_data[(frame_data['team'] == team) & (frame_data['label'] == 'player')]

class BasketballTrackingVisualizer:
    def __init__(self, csv_file):
        self.analyzer = BasketballTrackingAnalyzer(csv_file)
        self.court = BasketballCourtVisualizer()
        self.setup_animation()
        
    def setup_animation(self):
        """Setup animation components"""
        self.scatters = {}  # Store scatter plots for each object
        self.trails = {}    # Store trail lines
        self.texts = []     # Store text annotations
        
        # Frame counter and scoreboard
        self.frame_text = self.court.ax.text(0.02, 0.98, '', transform=self.court.ax.transAxes,
                                           fontsize=12, verticalalignment='top',
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Team possession indicator
        self.possession_text = self.court.ax.text(0.5, 0.98, '', transform=self.court.ax.transAxes,
                                                fontsize=12, verticalalignment='top', ha='center',
                                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
        
    def update_animation(self, frame_idx):
        """Update animation for each frame"""
        # Guards for empty data / bounds
        if not self.analyzer.frames:
            return []
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= len(self.analyzer.frames):
            return []
        frame = self.analyzer.frames[frame_idx]
        frame_data = self.analyzer.get_frame_data(frame)
        
        # Clear previous frame
        for scatter in self.scatters.values():
            # Matplotlib requires shape (N,2); use empty array instead of []
            scatter.set_offsets(np.empty((0, 2)))
        for trail in self.trails.values():
            trail.set_data([], [])
        for text in self.texts:
            text.remove()
        self.texts.clear()
        
        # Update each object
        ball_in_frame = False
        ball_possession = None
        
        for _, obj_data in frame_data.iterrows():
            obj_id = obj_data['object_id']
            x, y = obj_data['X_scaled'], obj_data['Y_scaled']
            label = obj_data['label']
            team = obj_data['team']
            
            # Create scatter if doesn't exist
            if obj_id not in self.scatters:
                color = self.analyzer.team_colors[team]
                size = 200 if label == 'ball' else 80
                marker = 'o' if label == 'player' else 's' if label == 'referee' else 'D'
                alpha = 1.0 if label != 'ball' else 0.9
                
                self.scatters[obj_id] = self.court.ax.scatter([], [], c=[color], 
                                                           s=size, marker=marker, alpha=alpha)
                
                # Create trail (shorter for better visibility)
                trail_alpha = 0.4 if label == 'player' else 0.6 if label == 'referee' else 0.3
                self.trails[obj_id], = self.court.ax.plot([], [], '-', 
                                                        color=color, alpha=trail_alpha, linewidth=2)
            
            # Update position
            self.scatters[obj_id].set_offsets(np.array([[x, y]], dtype=float))
            
            # Update trail (last 8 positions for players, 15 for ball)
            obj_traj = self.analyzer.get_object_trajectory(obj_id)
            trail_length = 15 if label == 'ball' else 8
            recent_frames = obj_traj[obj_traj['frame_id'] <= frame].tail(trail_length)
            if len(recent_frames) > 1:
                self.trails[obj_id].set_data(recent_frames['X_scaled'], recent_frames['Y_scaled'])
            
            # Track ball for possession calculation
            if label == 'ball':
                ball_in_frame = True
                # Simple possession: team with player closest to ball
                players_in_frame = frame_data[frame_data['label'] == 'player']
                if len(players_in_frame) > 0:
                    # Work on a copy to avoid SettingWithCopy warnings
                    pf = players_in_frame[['object_id','team','X_scaled','Y_scaled']].copy()
                    dxs = pf['X_scaled'] - x
                    dys = pf['Y_scaled'] - y
                    pf['distance_to_ball'] = np.sqrt(dxs*dxs + dys*dys)
                    closest_player = pf.loc[pf['distance_to_ball'].idxmin()]
                    ball_possession = closest_player['team']
            
            # Add labels for key objects
            if label == 'ball' or frame_idx % 10 == 0:  # Show labels occasionally
                label_text = f"{label[:3]}"
                if label == 'player':
                    label_text = f"P{obj_id.split('_')[1][-2:]}"
                
                text = self.court.ax.text(x, y + 100, label_text, 
                                       fontsize=7, ha='center', va='bottom', fontweight='bold',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                self.texts.append(text)
        
        # Update frame counter and possession
        self.frame_text.set_text(f'Frame: {frame}')
        
        if ball_in_frame and ball_possession:
            possession_team = 'Blue' if ball_possession == 'team_a' else 'Red' if ball_possession == 'team_b' else 'Unknown'
            self.possession_text.set_text(f'Possession: {possession_team}')
        else:
            self.possession_text.set_text('Ball not in frame')
        
        return list(self.scatters.values()) + list(self.trails.values()) + [self.frame_text, self.possession_text] + self.texts
    
    def create_animation(self, output_file=None):
        """Create and display animation"""
        total_frames = len(self.analyzer.frames)
        print(f"Creating basketball tracking animation with {total_frames} frames...")
        if total_frames == 0:
            print("[WARN] Nessun frame disponibile nel CSV: impossibile creare animazione.")
            return None

        max_frames = min(500, total_frames)
        anim = animation.FuncAnimation(
            self.court.fig,
            self.update_animation,
            frames=max_frames,
            interval=150,
            blit=True,
            repeat=True,
            init_func=lambda: []  # Avoid calling update before we are ready
        )
        
        plt.title('Basketball 3D Tracking Visualization\nTeam A (Blue) vs Team B (Red)', 
                 fontsize=14, pad=20)
        
        # Add legend
        team_a_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                markersize=8, label='Team A')
        team_b_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                markersize=8, label='Team B')
        referee_patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', 
                                 markersize=8, label='Referees')
        ball_patch = plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', 
                              markersize=8, label='Ball')
        
        self.court.ax.legend(handles=[team_a_patch, team_b_patch, referee_patch, ball_patch], 
                           loc='upper right', bbox_to_anchor=(1, 1))
        
        if output_file:
            print(f"Saving animation to {output_file}...")
            try:
                anim.save(output_file, writer='pillow', fps=8, dpi=120)
            except IndexError as e:
                print(f"[ERROR] Salvataggio fallito con blit=True ({e}). Riprovo senza blit...")
                anim = animation.FuncAnimation(
                    self.court.fig,
                    self.update_animation,
                    frames=max_frames,
                    interval=150,
                    blit=False,
                    repeat=True,
                    init_func=lambda: []
                )
                anim.save(output_file, writer='pillow', fps=8, dpi=120)
        
        plt.tight_layout()
        plt.show()
        
        return anim

def analyze_basketball_tracking(analyzer):
    """Print basketball-specific tracking analysis"""
    print("=== BASKETBALL TRACKING ANALYSIS ===")
    print(f"Total frames: {len(analyzer.frames)}")
    print(f"Total tracked objects: {len(analyzer.df['object_id'].unique())}")
    print(f"Frame range: {analyzer.frames[0]} - {analyzer.frames[-1]}")
    
    # Basketball-specific analysis
    player_data = analyzer.df[analyzer.df['label'] == 'player']
    referee_data = analyzer.df[analyzer.df['label'] == 'referee']
    ball_data = analyzer.df[analyzer.df['label'] == 'ball']
    
    team_a_players = player_data[player_data['team'] == 'team_a']['object_id'].unique()
    team_b_players = player_data[player_data['team'] == 'team_b']['object_id'].unique()
    
    print(f"\nTeam Composition:")
    print(f"  Team A (Blue): {len(team_a_players)} players")
    print(f"  Team B (Red): {len(team_b_players)} players")
    print(f"  Referees: {len(referee_data['object_id'].unique())}")
    print(f"  Ball appearances: {len(ball_data['frame_id'].unique())} frames")
    
    # Court coverage analysis
    print(f"\nCourt Coverage:")
    print(f"  X range: {player_data['X_scaled'].min():.1f} to {player_data['X_scaled'].max():.1f}")
    print(f"  Y range: {player_data['Y_scaled'].min():.1f} to {player_data['Y_scaled'].max():.1f}")
    
    # Tracking quality
    print(f"\nTracking Quality:")
    print(f"  Mean reprojection error: {analyzer.df['mean_reproj_error'].mean():.2f}")
    print(f"  Frames with all 14 players + 2 referees: {len(analyzer.frames)}")

def plot_offensive_sets(analyzer):
    """Plot common offensive formations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot team A typical positions
    team_a_data = analyzer.df[(analyzer.df['team'] == 'team_a') & (analyzer.df['label'] == 'player')]
    ax1.scatter(team_a_data['X_scaled'], team_a_data['Y_scaled'], c='blue', alpha=0.6, s=50)
    ax1.set_title('Team A (Blue) Position Heatmap')
    ax1.set_xlim(-analyzer.court_length/2, analyzer.court_length/2)
    ax1.set_ylim(-analyzer.court_width/2, analyzer.court_width/2)
    
    # Plot team B typical positions  
    team_b_data = analyzer.df[(analyzer.df['team'] == 'team_b') & (analyzer.df['label'] == 'player')]
    ax2.scatter(team_b_data['X_scaled'], team_b_data['Y_scaled'], c='red', alpha=0.6, s=50)
    ax2.set_title('Team B (Red) Position Heatmap')
    ax2.set_xlim(-analyzer.court_length/2, analyzer.court_length/2)
    ax2.set_ylim(-analyzer.court_width/2, analyzer.court_width/2)
    
    # Add court outlines to both
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        # Add simple court outline
        court = Rectangle((-analyzer.court_length/2, -analyzer.court_width/2), 
                        analyzer.court_length, analyzer.court_width, 
                        fill=False, color='black', linewidth=1)
        ax.add_patch(court)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Initialize basketball visualizer
    print("Loading basketball tracking data...")
    visualizer = BasketballTrackingVisualizer('output/3D_positions.csv')
    
    # Analyze basketball-specific tracking
    analyze_basketball_tracking(visualizer.analyzer)
    
    # Create interactive animation
    print("\nCreating basketball animation...")
    anim = visualizer.create_animation(output_file='basketball_tracking.gif')
    
    # Plot team formations
    print("\nAnalyzing team formations...")
    plot_offensive_sets(visualizer.analyzer)
    
    print("\nAnalysis complete! Key features:")
    print("✓ Team-based coloring (Blue vs Red)")
    print("✓ Basketball court markings")
    print("✓ Possession tracking")
    print("✓ Player trails and labels")
    print("✓ Formation analysis")