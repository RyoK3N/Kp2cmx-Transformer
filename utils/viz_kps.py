import matplotlib.pyplot as plt
import numpy as np
from dataset.skeleton import Skeleton

def visualize_predictions(outputs, targets, return_fig=False, title='Camera Matrix Comparison'):
    """
    Enhanced visualization of camera matrix predictions.
    
    Args:
        outputs: Predicted camera matrices (B, 16)
        targets: Ground truth camera matrices (B, 16)
        return_fig: If True, return the figure instead of displaying it
        title: Title for the plot
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
    
    # 1. Bar Plot Comparison
    ax1 = fig.add_subplot(gs[0, :])
    pred = outputs[0].reshape(-1)
    target = targets[0].reshape(-1)
    x = np.arange(len(pred))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pred, width, label='Predicted', color='blue', alpha=0.6)
    bars2 = ax1.bar(x + width/2, target, width, label='Ground Truth', color='red', alpha=0.6)
    
    ax1.set_title(f'{title}\nPredicted vs Ground Truth Values')
    ax1.set_xlabel('Matrix Elements')
    ax1.set_ylabel('Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', rotation=45, fontsize=8)
    
    # 2. Absolute Error Plot
    ax2 = fig.add_subplot(gs[1, 0])
    error = np.abs(pred - target)
    ax2.bar(x, error, color='red', alpha=0.6)
    ax2.set_title('Absolute Error')
    ax2.set_xlabel('Matrix Elements')
    ax2.set_ylabel('Absolute Error')
    ax2.grid(True, alpha=0.3)
    
    # Add error values
    for i, err in enumerate(error):
        ax2.text(i, err, f'{err:.3f}', ha='center', va='bottom', rotation=45, fontsize=8)
    
    # 3. Relative Error Plot
    ax3 = fig.add_subplot(gs[1, 1])
    relative_error = np.abs((pred - target) / (np.abs(target) + 1e-8)) * 100
    ax3.bar(x, relative_error, color='orange', alpha=0.6)
    ax3.set_title('Relative Error (%)')
    ax3.set_xlabel('Matrix Elements')
    ax3.set_ylabel('Relative Error %')
    ax3.grid(True, alpha=0.3)
    
    # Add relative error values
    for i, err in enumerate(relative_error):
        ax3.text(i, err, f'{err:.1f}%', ha='center', va='bottom', rotation=45, fontsize=8)
    
    # Add summary statistics
    stats_text = (
        f'Mean Absolute Error: {error.mean():.4f}\n'
        f'Max Absolute Error: {error.max():.4f}\n'
        f'Mean Relative Error: {relative_error.mean():.2f}%\n'
        f'Max Relative Error: {relative_error.max():.2f}%'
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        plt.close()

def visualize_keypoint_skeleton(keypoints, skeleton, return_fig=False):
    """
    Visualize the keypoint skeleton.
    
    Args:
        keypoints: Array of keypoint coordinates
        skeleton: Skeleton object defining the connections
        return_fig: If True, return the figure instead of displaying it
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # Plot keypoints
    keypoints_2d = keypoints.reshape(-1, 2)
    
    # Plot connections first (behind points)
    for child, parent in skeleton.connections:
        if child in skeleton.joint_indices and parent in skeleton.joint_indices:
            child_idx = skeleton.joint_indices[child]
            parent_idx = skeleton.joint_indices[parent]
            
            x = [keypoints_2d[child_idx, 0], keypoints_2d[parent_idx, 0]]
            y = [keypoints_2d[child_idx, 1], keypoints_2d[parent_idx, 1]]
            ax.plot(x, y, 'b-', alpha=0.6, linewidth=2)
    
    # Plot points on top
    ax.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], c='red', s=100, zorder=5)
    
    # Add joint labels with better positioning
    for joint, idx in skeleton.joint_indices.items():
        ax.annotate(joint, 
                   (keypoints_2d[idx, 0], keypoints_2d[idx, 1]),
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8,
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    ax.set_title('Keypoint Skeleton')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio and adjust limits
    ax.set_aspect('equal')
    
    # Add some padding around the skeleton
    x_min, x_max = keypoints_2d[:, 0].min(), keypoints_2d[:, 0].max()
    y_min, y_max = keypoints_2d[:, 1].min(), keypoints_2d[:, 1].max()
    padding = 0.2 * max(x_max - x_min, y_max - y_min)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    plt.tight_layout()
    
    if return_fig:
        return fig
    else:
        plt.show()
        plt.close()