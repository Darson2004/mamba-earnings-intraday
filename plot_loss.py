import matplotlib.pyplot as plt
import re
import numpy as np

def parse_log_file(log_file_path):
    """Parse the log.txt file to extract epoch and loss data."""
    epochs = []
    losses = []
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                # Look for lines that match the pattern "Epoch X/Y - Loss: Z" format
                # Example: "Epoch 1/3 - Loss: 20.7053"
                match = re.search(r'Epoch\s+(\d+)/\d+\s+-\s+Loss:\s+([\d.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    epochs.append(epoch)
                    losses.append(loss)
    except FileNotFoundError:
        print(f"Error: Could not find {log_file_path}")
        return [], []
    except Exception as e:
        print(f"Error reading log file: {e}")
        return [], []
    
    return epochs, losses

def plot_loss(epochs, losses, save_path='loss_plot.png'):
    """Plot the loss data."""
    if not epochs or not losses:
        print("No loss data found to plot.")
        return
    
    # Skip the first epoch (index 0)
    if len(epochs) > 1:
        epochs = epochs[1:]
        losses = losses[1:]
        print(f"Plotting {len(epochs)} epochs (excluding first epoch)")
    else:
        print("Only one epoch found, nothing to plot after excluding first epoch.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-o', linewidth=2, markersize=6, label='Training Loss')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Customize the plot
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time (Excluding First Epoch)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    
    # Add some statistics
    min_loss = min(losses)
    max_loss = max(losses)
    final_loss = losses[-1]
    
    # Add text box with statistics
    stats_text = f'Min Loss: {min_loss:.6f}\nMax Loss: {max_loss:.6f}\nFinal Loss: {final_loss:.6f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Loss plot saved as '{save_path}'")
    print(f"Total epochs plotted: {len(epochs)} (excluding first epoch)")
    print(f"Loss range: {min_loss:.6f} to {max_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")

def main():
    """Main function to parse log and plot loss."""
    log_file = 'log.txt'
    
    print("Parsing log file...")
    epochs, losses = parse_log_file(log_file)
    
    if epochs and losses:
        print(f"Found {len(epochs)} epochs of loss data")
        plot_loss(epochs, losses)
    else:
        print("No loss data found in log.txt")

if __name__ == "__main__":
    main() 