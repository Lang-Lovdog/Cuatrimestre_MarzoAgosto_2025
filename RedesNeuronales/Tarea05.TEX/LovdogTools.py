import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

def l1l2_interpretation():
  # Set up the figure and axis
  fig, ax = plt.subplots(1, 1, figsize=(7, 7))
  
  # 1. Create a loss function contour (e.g., MSE)
  w1 = np.linspace(-1.5, 1.5, 100)
  w2 = np.linspace(-1.5, 1.5, 100)
  W1, W2 = np.meshgrid(w1, w2)
  
  # A simple quadratic loss with its minimum off-center
  Loss = (W1 - 0.8)**2 + (W2 - 0.6)**2
  
  # Plot the contours of the loss function
  contour_levels = np.linspace(0.2, 2.5, 8)
  CS = ax.contour(W1, W2, Loss, levels=contour_levels, alpha=0.7, cmap='viridis')
  ax.clabel(CS, inline=1, fontsize=10)
  ax.plot(0.8, 0.6, 'ko', markersize=8, label='Optimal (Unregularized)')
  
  # 2. Draw the Constraint Regions ("The Budget")
  # L2 Constraint: ||w||₂² <= t (Circle)
  l2_constraint = Circle((0, 0), radius=1.0, fill=False, color='red', linestyle='--', linewidth=3, label=r'L₂ Constraint ($||\mathbf{w}||_2^2 \leq t$)')
  ax.add_patch(l2_constraint)
  
  # L1 Constraint: ||w||₁ <= t (Diamond) |w1| + |w2| <= t
  # Define the vertices of the diamond: (1,0), (0,1), (-1,0), (0,-1)
  diamond_vertices = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
  l1_constraint = Polygon(diamond_vertices, closed=True, fill=False, color='blue', linestyle='--', linewidth=3, label=r'L₁ Constraint ($||\mathbf{w}||_1 \leq t$)')
  ax.add_patch(l1_constraint)
  
  # 3. Find and mark the solutions where the contour touches the constraints
  # The regularized solution is the point on the constraint region where the
  # loss contour is tangent to it.
  
  # For visualization, we approximate these points.
  ax.plot(0.6, 0.8, 'ro', markersize=8, label='L₂ (Ridge) Solution') # Point on circle where contour is tangent
  ax.plot(1.0, 0.0, 'bo', markersize=8, label='L₁ (Lasso) Solution') # Point on diamond where contour is tangent
  
  # Draw lines from origin to the L1 solution to show sparsity (one weight is zero)
  ax.arrow(0, 0, 1.0, 0.0, head_width=0.05, head_length=0.05, fc='blue', ec='blue', linestyle=':')
  
  # Add labels and legend
  ax.set_xlabel(r'Weight $w_1$', fontsize=12)
  ax.set_ylabel(r'Weight $w_2$', fontsize=12)
  ax.set_title('Geometric Interpretation of L\u2081 vs. L\u2082 Regularization', fontsize=14)
  ax.legend(loc='upper right', fontsize=10)
  ax.grid(True, alpha=0.3)
  ax.set_aspect('equal')  # Essential for the circles and diamonds to look correct
  ax.set_xlim(-1.5, 1.5)
  ax.set_ylim(-1.5, 1.5)
  
  plt.tight_layout()
  plt.show()
