import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from dotenv import load_dotenv
import argparse
from utils.sanity_checks import sanity_check
from torch.utils.tensorboard import SummaryWriter
from model.model2 import KTPFormerV2
import datetime
import math
import matplotlib.pyplot as plt
from torchviz import make_dot
import torchvision
# Local imports
from model.model import KTPFormer
from dataset.mocap_dataset import MocapDataset
from dataset.skeleton import Skeleton
from utils.graph_utils import adj_mx_from_skeleton
from model.loss import weighted_frobenius_loss
from model.weights import initialize_weights
from utils.viz_kps import visualize_predictions, visualize_keypoint_skeleton
def parse_args():
    parser = argparse.ArgumentParser(description='Training script for KTPFormer')
    
    # Training hyperparameters
    parser.add_argument('--random_seed', type=int, default=100,
                        help='Random seed for reproducibility')
    parser.add_argument('--data_fraction', type=float, default=0.001,
                        help='Fraction of data to use for training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of epochs for learning rate warmup')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    # Model saving and data splits
    parser.add_argument('--model_save_path', type=str, 
                        default='./weights/ktpformer_best_model.pth',
                        help='Path to save the model')
    parser.add_argument('--train_size', type=float, default=0.7,
                        help='Fraction of data to use for training')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    
    # Training configuration
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--visualize_every', type=int, default=1,
                        help='Visualize every N epochs')
    
    # Add tensorboard logging argument
    parser.add_argument('--log_dir', type=str, default='runs',
                        help='Directory for tensorboard logs')

    args = parser.parse_args()
    return args

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def visualize_model_graph(model, writer, input_size=(1, 31, 2)):
    """
    Visualize model architecture and activations in TensorBoard.
    """
    try:
        # Create dummy input
        dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
        
        # Add graph to tensorboard
        writer.add_graph(model, dummy_input)
        writer.flush()
        
        # Add model summary as text
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_summary = (
            f"Model Summary:\n"
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,}\n"
            f"Input shape: {input_size}\n"
        )
        writer.add_text('Model/Summary', model_summary)
        
        # Track initial parameter distributions
        for name, param in model.named_parameters():
            writer.add_histogram(f'Parameters/{name}', param.data, 0)
            
    except Exception as e:
        print(f"Warning: Failed to visualize model graph: {str(e)}")

def train(args):
    # Set seeds for reproducibility
    set_seeds(args.random_seed)

    # Initialize tensorboard writer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(args.log_dir, current_time)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # Load environment variables
    load_dotenv()
    uri = os.getenv('URI')
    if not uri:
        raise EnvironmentError("Please set the 'URI' environment variable in your .env file.")

    # Define skeleton structure
    connections = [
        ('Head', 'Neck'), ('Neck', 'Chest'), ('Chest', 'Hips'),
        ('Neck', 'LeftShoulder'), ('LeftShoulder', 'LeftArm'),
        ('LeftArm', 'LeftForearm'), ('LeftForearm', 'LeftHand'),
        ('Chest', 'RightShoulder'), ('RightShoulder', 'RightArm'),
        ('RightArm', 'RightForearm'), ('RightForearm', 'RightHand'),
        ('Hips', 'LeftThigh'), ('LeftThigh', 'LeftLeg'),
        ('LeftLeg', 'LeftFoot'), ('Hips', 'RightThigh'),
        ('RightThigh', 'RightLeg'), ('RightLeg', 'RightFoot'),
        ('RightHand', 'RightFinger'), ('RightFinger', 'RightFingerEnd'),
        ('LeftHand', 'LeftFinger'), ('LeftFinger', 'LeftFingerEnd'),
        ('Head', 'HeadEnd'), ('RightFoot', 'RightHeel'),
        ('RightHeel', 'RightToe'), ('RightToe', 'RightToeEnd'),
        ('LeftFoot', 'LeftHeel'), ('LeftHeel', 'LeftToe'),
        ('LeftToe', 'LeftToeEnd'),
        ('SpineLow', 'Hips'), ('SpineMid', 'SpineLow'), ('Chest', 'SpineMid')
    ]

    joints_left = [
        'LeftShoulder', 'LeftArm', 'LeftForearm', 'LeftHand', 'LeftFinger', 'LeftFingerEnd',
        'LeftThigh', 'LeftLeg', 'LeftFoot', 'LeftHeel', 'LeftToe', 'LeftToeEnd'
    ]

    joints_right = [
        'RightShoulder', 'RightArm', 'RightForearm', 'RightHand', 'RightFinger', 'RightFingerEnd',
        'RightThigh', 'RightLeg', 'RightFoot', 'RightHeel', 'RightToe', 'RightToeEnd'
    ]

    # Initialize dataset
    dataset = MocapDataset(uri=uri, db_name='ai', collection_name='cameraPoses', skeleton=None)
    
    # Setup skeleton
    skeleton = Skeleton(
        connections=connections,
        joints_left=joints_left,
        joints_right=joints_right,
        ordered_joint_names=dataset.joint_names
    )
    dataset.skeleton = skeleton

    # Apply data fraction
    total_samples = len(dataset)
    samples_to_use = int(total_samples * args.data_fraction)
    dataset._ids = dataset._ids[:samples_to_use]
    dataset.total = samples_to_use

    print(f"Using {samples_to_use} samples out of {total_samples}")
    print(f"Number of joints: {dataset.num_joints}")
    print(f"Joint names: {dataset.joint_names}")

    sanity_check(dataset)
    
    split_generator = torch.Generator().manual_seed(args.random_seed)

    train_length = int(args.train_size * len(dataset))
    val_length = int(args.val_size * len(dataset))
    test_length = len(dataset) - train_length - val_length

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_length, val_length, test_length],
        generator=split_generator
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    loader_generator = torch.Generator().manual_seed(args.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  
        pin_memory=True,
        generator=loader_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )

    adj_matrix = adj_mx_from_skeleton(skeleton)
    model = KTPFormerV2(
        input_dim=dataset.num_joints * 2,
        embed_dim=512,
        adj=adj_matrix,
        depth=4,
        num_heads=8,
        drop_rate=0.2
    ).to(args.device)

    model.apply(initialize_weights)

    # Add model graph visualization
    visualize_model_graph(model, writer, input_size=(1, dataset.num_joints, 2))

    # Track gradients and weights
    for name, param in model.named_parameters():
        writer.add_histogram(f'Parameters/{name}', param.data, 0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    scaler = GradScaler() if args.device == 'cuda' else None

    def validate(epoch, show_visualization=True, skeleton=skeleton):
        model.eval()
        val_loss = 0.0
        batch_predictions = []
        batch_targets = []
        batch_inputs = []
        all_errors = []

        try:
            with torch.no_grad():
                for i, (keypoints, camera_matrix,_,_) in enumerate(val_loader):
                    keypoints = keypoints.to(args.device)
                    camera_matrix = camera_matrix.to(args.device)
                    
                    with torch.amp.autocast(device_type='cuda', enabled=(args.device == 'cuda')):
                        outputs = model(keypoints)
                        loss = weighted_frobenius_loss(outputs, camera_matrix)
                    
                    val_loss += loss.item()
                    
                    # Calculate errors for this batch
                    batch_error = torch.abs(outputs - camera_matrix)
                    all_errors.append(batch_error.cpu().numpy())

                    if show_visualization:
                        batch_predictions.append(outputs.detach().cpu().numpy())
                        batch_targets.append(camera_matrix.detach().cpu().numpy())
                        batch_inputs.append(keypoints.detach().cpu().numpy())

            val_loss /= len(val_loader)
            all_errors = np.concatenate(all_errors, axis=0)

            if show_visualization and batch_predictions:
                try:
                    # Visualize predictions for selected batches
                    for batch_idx in range(min(3, len(batch_predictions))):  # Show max 3 batches
                        # Prediction comparison
                        fig_pred = visualize_predictions(
                            batch_predictions[batch_idx], 
                            batch_targets[batch_idx], 
                            return_fig=True,
                            title=f'Batch {batch_idx} Predictions'
                        )
                        if fig_pred is not None:
                            writer.add_figure(
                                f'Predictions/batch_{batch_idx}', 
                                fig_pred, 
                                global_step=epoch
                            )
                        
                        # Skeleton visualization
                        fig_kps = visualize_keypoint_skeleton(
                            batch_inputs[batch_idx][0], 
                            skeleton=skeleton, 
                            return_fig=True
                        )
                        if fig_kps is not None:
                            writer.add_figure(
                                f'Keypoint_Skeleton/batch_{batch_idx}', 
                                fig_kps, 
                                global_step=epoch
                            )

                    # Add error distribution plot
                    fig_error_dist = plt.figure(figsize=(10, 5))
                    plt.hist(all_errors.flatten(), bins=50, alpha=0.7)
                    plt.title('Error Distribution')
                    plt.xlabel('Absolute Error')
                    plt.ylabel('Count')
                    writer.add_figure('Error_Distribution', fig_error_dist, global_step=epoch)
                    plt.close()

                    # Add error statistics
                    writer.add_scalar('Error/mean', np.mean(all_errors), epoch)
                    writer.add_scalar('Error/median', np.median(all_errors), epoch)
                    writer.add_scalar('Error/std', np.std(all_errors), epoch)
                    writer.add_scalar('Error/max', np.max(all_errors), epoch)
                    writer.add_scalar('Error/min', np.min(all_errors), epoch)

                except Exception as e:
                    print(f"Warning: Visualization error: {str(e)}")

            return val_loss
        except Exception as e:
            print(f"Error in validation: {str(e)}")
            return float('inf')

    best_val_loss = float('inf')
    no_improvement_count = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        try:
            for batch_idx, (keypoints, camera_matrix, _, _) in enumerate(progress_bar):
                keypoints = keypoints.to(args.device)
                camera_matrix = camera_matrix.to(args.device)
                
                optimizer.zero_grad()

                # Track activations periodically
                if batch_idx % 100 == 0:
                    model.track_activations = True
                
                with torch.amp.autocast(device_type='cuda', enabled=(args.device == 'cuda')):
                    if model.track_activations:
                        outputs, activations = model(keypoints)
                    else:
                        outputs = model(keypoints)
                    loss = weighted_frobenius_loss(outputs, camera_matrix)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Log activations and parameters
                if batch_idx % 100 == 0:
                    # Log parameter gradients and weights
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
                        writer.add_histogram(f'Parameters/{name}', param.data, global_step)
                    
                    # Log activations
                    if model.track_activations:
                        for name, activation in activations.items():
                            if activation is not None:
                                writer.add_histogram(f'Activations/{name}', 
                                                   activation.flatten(),
                                                   global_step)
                                
                                # Log mean and std of activations
                                writer.add_scalar(f'Activations/{name}_mean', 
                                               activation.mean().item(),
                                               global_step)
                                writer.add_scalar(f'Activations/{name}_std', 
                                               activation.std().item(),
                                               global_step)
                    
                    model.track_activations = False

                # Log statistics and visualizations
                if batch_idx % 50 == 0:  # Increased frequency of visualization
                    # Log values
                    writer.add_scalar('Stats/max_value', torch.max(camera_matrix).item(), global_step)
                    writer.add_scalar('Stats/min_value', torch.min(camera_matrix).item(), global_step)
                    
                    try:
                        # Visualize input skeleton
                        fig_kps = visualize_keypoint_skeleton(
                            keypoints[0].cpu().numpy(),
                            skeleton=skeleton,
                            return_fig=True
                        )
                        if fig_kps is not None:
                            writer.add_figure(
                                'Training/Input_Skeleton',
                                fig_kps,
                                global_step=global_step
                            )
                        plt.close(fig_kps)

                        # Visualize predictions vs targets
                        fig_pred = visualize_predictions(
                            outputs.detach().cpu().numpy(),
                            camera_matrix.detach().cpu().numpy(),
                            return_fig=True,
                            title=f'Training Batch {batch_idx}'
                        )
                        if fig_pred is not None:
                            writer.add_figure(
                                'Training/Predictions',
                                fig_pred,
                                global_step=global_step
                            )
                        plt.close(fig_pred)

                    except Exception as e:
                        print(f"Warning: Training visualization error: {str(e)}")

                total_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
                
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                global_step += 1

                # Log parameter gradients and weights periodically
                if batch_idx % 100 == 0:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
                        writer.add_histogram(f'Parameters/{name}', param.data, global_step)

                    # Log layer activations for a sample batch
                    with torch.no_grad():
                        # Get intermediate activations (you'll need to modify your model to return these)
                        sample_activations = model(keypoints[:1])  # Use first sample
                        if isinstance(sample_activations, tuple):
                            outputs, activations = sample_activations
                            for layer_name, activation in activations.items():
                                if activation is not None:
                                    writer.add_histogram(f'Activations/{layer_name}', 
                                                       activation.flatten(),
                                                       global_step)

            scheduler.step()

            show_viz = (epoch % args.visualize_every == 0)
            val_loss = validate(epoch, show_visualization=show_viz, skeleton=skeleton)

            # Visualize a grid of input skeletons at the end of each epoch
            if show_viz:
                try:
                    # Create a grid of skeletons from the last batch
                    fig = plt.figure(figsize=(20, 10))
                    num_samples = min(4, keypoints.size(0))
                    
                    for i in range(num_samples):
                        plt.subplot(2, 2, i+1)
                        keypoints_i = keypoints[i].cpu().numpy()
                        
                        # Plot connections
                        keypoints_2d = keypoints_i.reshape(-1, 2)
                        for child, parent in skeleton.connections:
                            if child in skeleton.joint_indices and parent in skeleton.joint_indices:
                                child_idx = skeleton.joint_indices[child]
                                parent_idx = skeleton.joint_indices[parent]
                                x = [keypoints_2d[child_idx, 0], keypoints_2d[parent_idx, 0]]
                                y = [keypoints_2d[child_idx, 1], keypoints_2d[parent_idx, 1]]
                                plt.plot(x, y, 'b-', alpha=0.6, linewidth=2)
                        
                        # Plot points
                        plt.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], c='red', s=50)
                        plt.title(f'Sample {i+1}')
                        plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    writer.add_figure('Training/Skeleton_Grid', fig, global_step=epoch)
                    plt.close(fig)
                except Exception as e:
                    print(f"Warning: Skeleton grid visualization error: {str(e)}")

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{args.epochs} Summary:")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")

            # Log epoch metrics
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
                torch.save(model.state_dict(), args.model_save_path)
                print(f"  Model saved with validation loss: {best_val_loss:.4f}")
            else:
                no_improvement_count += 1
                if no_improvement_count >= args.early_stop_patience:
                    print(f"No improvement for {args.early_stop_patience} validation checks. Early stopping.")
                    break

        except Exception as e:
            print(f"Error in training epoch {epoch+1}: {str(e)}")
            continue

    writer.close()

if __name__ == "__main__":
    args = parse_args()
    train(args)



