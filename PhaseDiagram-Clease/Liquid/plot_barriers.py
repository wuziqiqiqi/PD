import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re

# Set the directory containing the image files
directory = sys.argv[2]

# Initialize an empty matrix
N = int(sys.argv[1])  # Adjust this based on your grid size
matrix = np.zeros((2, 2, N, N))
assignmentCount = np.zeros((2, N, N))

# Function to parse the filename and update the matrix
def parse_filename(filename):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    assert re.match(r'^\d+_\d+_\d+(\.\d+)?_\d+(\.\d+)?$', name), "path contains other files! you idiot!"
    x, y, a, b = map(float, name.split('_'))
    x, y = int(x), int(y)

    # Handle periodic boundary conditions
    if abs(x - y) == N-1 or abs(x - y) == N*(N-1):
        x_row, x_col = divmod(max(x, y), N)
    else:
        x_row, x_col = divmod(min(x, y), N)

    if abs(x - y) == 1 and (min(x,y)+1)%N != 0:  # Horizontal connection
        # print(x,y, f"[0, {x_row}, {x_col}]", assignmentCount[0, x_row, x_col])
        if assignmentCount[0, x_row, x_col] > 1:
            assert matrix[0, 0, x_row, x_col] == a and matrix[0, 1, x_row, x_col] == b, f"same connection between {x} and {y}, different value! you idiot!"
        matrix[0, 0, x_row, x_col] = a
        matrix[0, 1, x_row, x_col] = b
        assignmentCount[0, x_row, x_col] += 1
    elif abs(x - y) == N-1 and min(x, y)%N == 0:
        # print(x,y, f"[0, {x_row}, {x_col}]", assignmentCount[0, x_row, x_col])
        if assignmentCount[0, x_row, x_col] > 1:
            assert matrix[0, 0, x_row, x_col] == b and matrix[0, 1, x_row, x_col] == a, f"same connection between {x} and {y}, different value! you idiot!"
        matrix[0, 0, x_row, x_col] = b
        matrix[0, 1, x_row, x_col] = a
        assignmentCount[0, x_row, x_col] += 1
    elif abs(x - y) == N:  # Vertical connection
        # print(x,y, f"[1, {x_row}, {x_col}]", assignmentCount[1, x_row, x_col])
        if assignmentCount[1, x_row, x_col] > 1:
            assert matrix[1, 0, x_row, x_col] == b and matrix[1, 1, x_row, x_col] == a, f"same connection between {x} and {y}, different value! you idiot!"
        matrix[1, 0, x_row, x_col] = b
        matrix[1, 1, x_row, x_col] = a
        assignmentCount[1, x_row, x_col] += 1
    elif abs(x - y) == N*(N-1):
        # print(x,y, f"[1, {x_row}, {x_col}]", assignmentCount[1, x_row, x_col])
        if assignmentCount[1, x_row, x_col] > 1:
            assert matrix[1, 0, x_row, x_col] == a and matrix[1, 1, x_row, x_col] == b, f"same connection between {x} and {y}, different value! you idiot!"
        matrix[1, 0, x_row, x_col] = a
        matrix[1, 1, x_row, x_col] = b
        assignmentCount[1, x_row, x_col] += 1
    else:
        # assert False, "impossible connectivity found! you idiot!"
        print("impossible connectivity found! you idiot!")


# Loop through all files in the directory
for filename in [os.path.join(dp, f) for dp, dn, fn in os.walk(directory) for f in fn]:
    if filename.endswith('.png') and "barriers.png" not in filename:
        parse_filename(os.path.join(directory, filename))
        
        
# Create a figure and axis
fig, ax = plt.subplots()

# Set up the grid
points = np.arange(N * N).reshape(N, N)

offset = 0.05
shrinkage = 0.1

# Draw the arrows based on the matrix values
for i in range(N):
    for j in range(N):
        # Horizontal arrows
        right_val = matrix[0, 0, i, j]
        left_val = matrix[0, 1, i, j]
        
        ax.annotate("", xy=(j + 1, N - 1 - i + offset), xytext=(j, N - 1 - i + offset), 
                    arrowprops=dict(facecolor=plt.cm.viridis(right_val), shrink=shrinkage))
        ax.annotate("", xy=(j, N - 1 - i - offset), xytext=(j + 1, N - 1 - i - offset), 
                    arrowprops=dict(facecolor=plt.cm.viridis(left_val), shrink=shrinkage))
        
        
        # Vertical arrows
        up_val = matrix[1, 0, i, j]
        down_val = matrix[1, 1, i, j]
        
        ax.annotate("", xy=(j - offset, N - 1 - i), xytext=(j - offset, N - 1 - i - 1), 
                    arrowprops=dict(facecolor=plt.cm.viridis(up_val), shrink=shrinkage))
        ax.annotate("", xy=(j + offset, N - 1 - i - 1), xytext=(j + offset, N - 1 - i), 
                    arrowprops=dict(facecolor=plt.cm.viridis(down_val), shrink=shrinkage))

    
# Set the aspect of the plot to be equal
ax.set_aspect('equal')

# Set axis limits
ax.set_xlim(-0.5, N)
ax.set_ylim(-1, N-0.5)

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Add grid
ax.grid(True)

# Show the plot
plt.savefig(os.path.join(directory, "barriers.png"))