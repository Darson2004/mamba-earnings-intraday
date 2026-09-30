#!/usr/bin/env python3
"""
Script to analyze and visualize the test results from test_results.npz
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(filename="test_results.npz"):
    """Load the test results from npz file."""
    try:
        results = np.load(filename)
        return results
    except FileNotFoundError:
        print(f"Results file {filename} not found. Please run test.py first.")
        return None

def analyze_results(results):
    """Analyze and visualize the test results."""
    mse = results['mse']
    confidence = results['confidence']
    prediction_counts = results['prediction_counts']
    time_indices = results['time_indices']
    
    print("="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    # Overall statistics
    valid_mse = mse[~np.isnan(mse)]
    valid_confidence = confidence[~np.isnan(confidence)]
    
    print(f"Total valid predictions: {len(valid_mse)}")
    print(f"Overall MSE: {np.mean(valid_mse):.4f} ± {np.std(valid_mse):.4f}")
    print(f"Overall 95% CI Coverage: {np.mean(valid_confidence):.3f} ({np.mean(valid_confidence)*100:.1f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. MSE heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(mse, aspect='auto', cmap='viridis')
    ax1.set_xlabel('Forecast Step (minutes)')
    ax1.set_ylabel('History Length')
    ax1.set_title('MSE by History Length and Forecast Step')
    ax1.set_yticks(range(0, len(time_indices), 30))
    ax1.set_yticklabels([time_indices[i] for i in range(0, len(time_indices), 30)])
    plt.colorbar(im1, ax=ax1, label='MSE')
    
    # 2. Confidence coverage heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(confidence, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    ax2.set_xlabel('Forecast Step (minutes)')
    ax2.set_ylabel('History Length')
    ax2.set_title('95% CI Coverage by History Length and Forecast Step')
    ax2.set_yticks(range(0, len(time_indices), 30))
    ax2.set_yticklabels([time_indices[i] for i in range(0, len(time_indices), 30)])
    plt.colorbar(im2, ax=ax2, label='Coverage Ratio')
    
    # 3. MSE vs History Length (averaged across forecast steps)
    ax3 = axes[1, 0]
    mse_by_history = np.nanmean(mse, axis=1)
    ax3.plot(time_indices, mse_by_history, 'b-o', linewidth=2, markersize=6)
    ax3.set_xlabel('History Length')
    ax3.set_ylabel('Average MSE')
    ax3.set_title('MSE vs History Length')
    ax3.grid(True, alpha=0.3)
    
    # 4. Confidence Coverage vs History Length (averaged across forecast steps)
    ax4 = axes[1, 1]
    conf_by_history = np.nanmean(confidence, axis=1)
    ax4.plot(time_indices, conf_by_history, 'r-o', linewidth=2, markersize=6)
    ax4.axhline(y=0.95, color='k', linestyle='--', alpha=0.7, label='Target 95%')
    ax4.set_xlabel('History Length')
    ax4.set_ylabel('Average 95% CI Coverage')
    ax4.set_title('Confidence Coverage vs History Length')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_results_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("DETAILED STATISTICS BY HISTORY LENGTH")
    print("="*60)
    
    for i, history_len in enumerate(time_indices[::30]):  # Every 30th index
        if i < len(mse_by_history):
            history_mse = mse_by_history[i*30] if i*30 < len(mse_by_history) else np.nan
            history_conf = conf_by_history[i*30] if i*30 < len(conf_by_history) else np.nan
            
            if not np.isnan(history_mse):
                print(f"History {history_len:3d}: MSE={history_mse:.4f}, CI Coverage={history_conf:.3f} ({history_conf*100:.1f}%)")
    
    # Forecast step analysis
    print("\n" + "="*60)
    print("DETAILED STATISTICS BY FORECAST STEP")
    print("="*60)
    
    mse_by_step = np.nanmean(mse, axis=0)
    conf_by_step = np.nanmean(confidence, axis=0)
    
    for step in range(0, 60, 10):  # Every 10th step
        if step < len(mse_by_step):
            step_mse = mse_by_step[step]
            step_conf = conf_by_step[step]
            print(f"Step {step:2d}: MSE={step_mse:.4f}, CI Coverage={step_conf:.3f} ({step_conf*100:.1f}%)")

def main():
    """Main function to run the analysis."""
    results = load_results()
    if results is not None:
        analyze_results(results)
    else:
        print("Please run test.py first to generate results.")

if __name__ == "__main__":
    main() 