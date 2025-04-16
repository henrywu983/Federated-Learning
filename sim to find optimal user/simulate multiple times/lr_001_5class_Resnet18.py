import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import time
import pandas as pd
import argparse
import os
from datetime import datetime
import sys
import random as rnd
import matplotlib.pyplot as plt
from collections import Counter
import random as rnd
from sklearn.metrics import confusion_matrix, f1_score
from FL_setting_NeurIPS import FederatedLearning

# Start time
start_time = time.time()

# Simulate command-line arguments
sys.argv = [
    'placeholder_script_name',
    '--learning_rate', '0.001',
    '--epochs', '3',
    '--batch_size', '64',
    '--num_users', '10',
    '--fraction', '0.05',
    '--transmission_probability', '0.1',
    '--num_slots', '10',
    '--num_timeframes', '148',
    '--user_data_size', '1500',
    '--seeds', '56', '3', '29', '85', '65',
    '--gamma_momentum', '0',
    '--use_memory_matrix', 'false',
    '--arrival_rate', '0.1',
    '--phase', '15', # number of timeframes per phase, there are in total five phases
    '--num_runs', '5',
    '--slotted_aloha', 'false', # for the NeurIPS paper, we don't consider random access channel
    '--num_memory_cells', '4',
    '--selected_mode', 'vanilla',
    '--cycle', '2'
]

# Command-line arguments
parser = argparse.ArgumentParser(description="Federated Learning with Slotted ALOHA and CIFAR-10 Dataset")
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--num_users', type=int, default=10, help='Number of users in federated learning')
parser.add_argument('--fraction', type=float, nargs='+', default=[0.1], help='Fraction for top-k sparsification')
parser.add_argument('--transmission_probability', type=float, default=0.1, help='Transmission probability for Slotted ALOHA')
parser.add_argument('--num_slots', type=int, default=10, help='Number of slots for Slotted ALOHA simulation')
parser.add_argument('--num_timeframes', type=int, default=15, help='Number of timeframes for simulation')
parser.add_argument('--seeds', type=int, nargs='+', default=[85, 12, 29], help='Random seeds for averaging results')
parser.add_argument('--gamma_momentum', type=float, nargs='+', default=[0.6], help='Momentum for memory matrix')
parser.add_argument('--use_memory_matrix', type=str, default='true', help='Switch to use memory matrix (true/false)')
parser.add_argument('--user_data_size', type=int, default=2000, help='Number of samples each user gets')
parser.add_argument('--arrival_rate', type=float, default=0.5,help='Arrival rate of new information')
parser.add_argument('--phase', type=int, default=5,help='When concept drift happens, when distribution change from one Class to another')
parser.add_argument('--num_runs', type=int, default=5,help='Number of simulations')
parser.add_argument('--slotted_aloha', type=str, default='true',help='Whether we use Slotted aloha in the simulation')
parser.add_argument('--num_memory_cells', type=int, default=6,help='Number of memory cells per client')
parser.add_argument('--selected_mode', type=str, default='vanilla',help='Which setting we are using: genie_aided, vanilla, user_selection')
parser.add_argument('--cycle', type=int, default=1,help='Number of cycles')

args = parser.parse_args()

# Parsed arguments
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
num_users = args.num_users
fraction = args.fraction
transmission_probability = args.transmission_probability
num_slots = args.num_slots
num_timeframes = args.num_timeframes
seeds_for_avg = args.seeds
gamma_momentum = args.gamma_momentum
use_memory_matrix = args.use_memory_matrix.lower() == 'true'
user_data_size = args.user_data_size
tx_prob = args.transmission_probability
arrival_rate = args.arrival_rate
phase = args.phase
num_runs = args.num_runs
slotted_aloha = args.slotted_aloha
num_memory_cells = args.num_memory_cells
selected_mode = args.selected_mode
cycle = args.cycle

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n{'*' * 50}\n*** Using device: {device} ***\n{'*' * 50}\n")

class_mappings = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
    4: [8, 9]
}

# Function to map original labels to new classes
def map_to_new_classes(original_labels):
    new_labels = np.zeros_like(original_labels)
    for new_class, original_classes in class_mappings.items():
        for original_class in original_classes:
            new_labels[original_labels == original_class] = new_class
    return new_labels

# CIFAR-10 dataset and preprocessing
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet18 expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
x_train, y_train = trainset.data, np.array(trainset.targets)

assert len(x_train) >= num_users * user_data_size, "Dataset too small for requested user allocation!"

# Resnet18 Model
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Remove the final fc layer
        self.classifier = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
# Sparsify the model weights
def top_k_sparsificate_model_weights(weights, fraction):
    flat_weights = torch.cat([w.view(-1) for w in weights])
    threshold_value = torch.quantile(torch.abs(flat_weights), 1 - fraction)
    new_weights = []
    for w in weights:
        mask = torch.abs(w) >= threshold_value
        new_weights.append(w * mask.float())
    return new_weights

# Simulate transmissions on the slotted ALOHA channel
def simulate_transmissions(num_users, transmission_probability):
    decisions = np.random.rand(num_users) < transmission_probability
    if np.sum(decisions) == 1:
        return [i for i, decision in enumerate(decisions) if decision]
    return []

# Calculate gradient difference between two sets of weights
def calculate_gradient_difference(w_before, w_after):
    return [w_after[k] - w_before[k] for k in range(len(w_after))]

# Partitioning User Data into Memory Cells
def partition_data_per_user(x_data, y_data, pattern, num_users, cell_size):
    """
    Partition x_data and y_data among num_users, each with memory cells as specified by pattern.
    Each memory cell will contain data exclusively from the class defined in the pattern.
    
    Args:
        x_data (np.array): Array of images.
        y_data (np.array): Array of labels (assumed to be the remapped new classes).
        pattern (list): List of ints specifying the class for each memory cell (e.g., [0, 2, 1, 1, 3, 4]).
        num_users (int): Number of users.
        cell_size (int): Fixed number of samples each memory cell should contain.
    
    Returns:
        dict: Mapping from user_id to a dictionary where each key is a memory cell index and the value is a dict
              containing 'x' (the data samples) and 'y' (the corresponding labels).
    """
    # Group indices of x_data by their class label
    # Ex: y_data = np.array([0, 2, 1, 1, 3, 4, 0, 2, 1, 1])
    # indices_by_class would look like:
    # {
    #     0: [0, 6],
    #     1: [2, 3, 8, 9],
    #     2: [1, 7],
    #     3: [4],
    #     4: [5]
    # }
    indices_by_class = {}
    unique_classes = np.unique(y_data)
    for cls in unique_classes:
        indices_by_class[cls] = np.where(y_data == cls)[0].tolist()
    
    # Shuffle indices for each class to ensure random sampling
    for cls in indices_by_class:
        rnd.shuffle(indices_by_class[cls])
    
    # Verify that enough samples exist for each class based on the pattern
    pattern_counts = Counter(pattern)
    for cls, count in pattern_counts.items():
        # Max{count} = num_memory_cell, so Max{required} <= 
        required = count * cell_size * num_users
        available = len(indices_by_class[cls])
        assert available >= required, (
            f"Not enough samples for class {cls}: required {required}, available {available}"
        )
    
    # Allocate data to each user's memory cells according to the pattern
    user_data = {}
    for user_id in range(num_users):
        user_data[user_id] = {}
        for cell_idx, cell_class in enumerate(pattern):
            selected_indices = indices_by_class[cell_class][:cell_size]
            indices_by_class[cell_class] = indices_by_class[cell_class][cell_size:]
            cell_x = x_data[selected_indices]
            cell_y = y_data[selected_indices]
            user_data[user_id][cell_idx] = {'x': cell_x, 'y': cell_y}
    
    return user_data

# Concept drift function
def apply_concept_drift(train_data_X, train_data_Y, num_users, x_train, y_train, arrival_rate, timeframe, cell_size, num_memory_cells, user_new_info_dict):

    # Calculate which phase we are in (aka what class should be injected)
    current_round_user_data_info = {user: False for user in range(num_users)}
    current_phase = (timeframe) // phase % len(class_mappings)
    print(f"Apply concept drift --> Phase {current_phase}, Inject Class {current_phase}")

    # For sampling new data from the global pool, get indices for the new_class
    global_indices = np.where(np.isin(y_train, class_mappings[current_phase]))[0]

    # Prepare new memory structures
    new_memory_X = {}
    new_memory_Y = {}

     # For each user, decide whether drift occurs based on arrival_rate.
    for user in range(num_users):
        # Split the aggregated data into memory cells.
        cells_X = np.split(train_data_X[user], num_memory_cells, axis=0)
        cells_Y = np.split(train_data_Y[user], num_memory_cells, axis=0)

        if np.random.rand() < arrival_rate:
            current_round_user_data_info[user] = True
            user_new_info_dict[user] += 1
            # move the data from cell (i-1) into cell i.
            for cell in range(num_memory_cells - 1, 0, -1):
                cells_X[cell] = cells_X[cell - 1]
                cells_Y[cell] = cells_Y[cell - 1]

            # Sample new data for memory cell 0 from the global pool, for the new_class.
            # If not enough samples are available, allow replacement.
            if len(global_indices) < cell_size:
                sampled_indices = np.random.choice(global_indices, cell_size, replace=True)
            else:
                sampled_indices = np.random.choice(global_indices, cell_size, replace=False)
            cells_X[0] = x_train[sampled_indices]
            cells_Y[0] = y_train[sampled_indices]    
            
        new_memory_X[user] = cells_X
        new_memory_Y[user] = cells_Y

    new_stale_data_info[run][seed_index][timeframe] = user_new_info_dict.copy()

    updated_train_data_X = {}
    updated_train_data_Y = {}
    for user in range(num_users):
        updated_train_data_X[user] = np.concatenate(new_memory_X[user], axis=0)
        updated_train_data_Y[user] = np.concatenate(new_memory_Y[user], axis=0)

    return updated_train_data_X, updated_train_data_Y, user_new_info_dict, current_round_user_data_info

def evaluate_per_class_accuracy(model, testloader, device, num_classes=5):
    """
    Evaluate and print the accuracy of the model on the test data for each class.
    
    Args:
        model (nn.Module): The neural network model.
        testloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to run the evaluation on.
    
    Returns:
        dict: A dictionary with the per-class accuracies.
    """
    with torch.no_grad():
        accuracies = {}
        total_correct = 0
        
        # Track total samples for each class
        class_counts = {i: 0 for i in range(num_classes)}
        class_correct = {i: 0 for i in range(num_classes)}

        for images, labels in testloader:
            images, labels = images.to(device), labels.numpy()
            
            # Map labels to new classes
            new_labels = map_to_new_classes(labels)
            new_labels = torch.tensor(new_labels).to(device)

            # Forward pass
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            # Map model predictions (still in 10-class space) to the new 5-class space
            mapped_predictions = map_to_new_classes(predictions.cpu().numpy())
            mapped_predictions = torch.tensor(mapped_predictions).to(device)

            for class_idx in range(num_classes):
                class_mask = (new_labels == class_idx)
                class_counts[class_idx] += class_mask.sum().item()
                class_correct[class_idx] += (mapped_predictions[class_mask] == class_idx).sum().item()

        # Compute overall accuracy by summing counts from all classes
        total_samples = sum(class_counts.values())
        total_correct = sum(class_correct.values())

        # Calculate accuracy for each class
        for class_idx in range(num_classes):
            if class_counts[class_idx] > 0:
                accuracies[class_idx] = 100 * class_correct[class_idx] / class_counts[class_idx]
            else:
                accuracies[class_idx] = 0  # If no samples exist, set accuracy to 0

            print(f"Accuracy for Class {class_idx}: {accuracies[class_idx]:.2f}%")

        # Compute overall accuracy as a percentage
        overall_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
    return accuracies, overall_accuracy

def evaluate_with_metrics(model, testloader, device, 
                          conf_matrix_dir='conf_matrices', f1_dir='f1_scores'):
    """
    Evaluate the model on the test set, compute a confusion matrix and F1 score,
    and save them to files for each timeframe.
    
    The function maps both the model's predictions and the ground truth labels from 
    the original 10 classes to 5 new classes using the map_to_new_classes function.
    
    Args:
        model (nn.Module): The neural network model.
        testloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the evaluation on.
        timeframe (int): The current timeframe identifier.
        num_classes (int): The number of new classes (default is 5).
        conf_matrix_dir (str): Directory where confusion matrix files will be saved.
        f1_dir (str): Directory where F1 score files will be saved.
    
    Returns:
        tuple: A tuple containing:
            - conf_matrix (np.ndarray): The confusion matrix.
            - f1 (float): The weighted F1 score across all classes.
    """
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels_np = labels.numpy()
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            # Map both predictions and labels from 10 classes to 5 classes
            mapped_preds = map_to_new_classes(preds.cpu().numpy())
            mapped_labels = map_to_new_classes(labels_np)
            
            all_preds.extend(mapped_preds)
            all_labels.extend(mapped_labels)
    
    # Convert lists to numpy arrays for metric computation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute confusion matrix and weighted F1 score
    conf_matrix = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    # Normalize the confusion matrix by rows (actual class)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_normalized = np.nan_to_num(conf_matrix_normalized)  # handles division by zero

    return conf_matrix_normalized, f1, f1_per_class


testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize matrices for results with an additional dimension for num_active_users
global_grad_mag = np.zeros((num_runs, len(seeds_for_avg), num_timeframes))

# Adjust other relevant matrices similarly
successful_users_record = np.zeros((num_runs, len(seeds_for_avg), num_timeframes))
loc_grad_mag = np.zeros((num_runs, len(seeds_for_avg), num_timeframes, num_users, num_users))
loc_grad_mag_memory = np.zeros((num_runs, len(seeds_for_avg), num_timeframes, num_users, num_users))
memory_matrix_mag = np.zeros((num_runs, len(seeds_for_avg), num_timeframes, num_users, num_users))

accuracy_distributions = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

accuracy_distributions_class_0 = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

accuracy_distributions_class_1 = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

accuracy_distributions_class_2 = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

accuracy_distributions_class_3 = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

accuracy_distributions_class_4 = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

conf_matrix_stats = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

f1_stats = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

f1_per_class_stats = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

correctly_received_packets_stats = {
    run: {
        seed_index: {
            timeframe: {'mean': None, 'variance': None}  # Removed num_active_users dimension
            for timeframe in range(num_timeframes)
        }
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

new_stale_data_info = {
    run: {
        seed_index: {timeframe: None for timeframe in range(num_timeframes)}
        for seed_index in range(len(seeds_for_avg))
    }
    for run in range(num_runs)
}

# Main training loop
seed_count = 1

for run in range(num_runs):
    rnd.seed(run)
    np.random.seed(run)
    torch.manual_seed(run)
    print(f"************ Run {run + 1} ************")

    # Define the initial memory pattern for all users:
    initial_pattern = [1, 2, 3, 4]
    
    # 12 memory cells
    # initial_pattern = [1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2]

    cell_size = user_data_size // num_memory_cells 

    # Partition the training data for each user using the defined pattern
    user_data_memory = partition_data_per_user(x_train, np.array(y_train),
                                               initial_pattern, num_users, cell_size)
    
    # Combine the memory cells for each user into one training set
    train_data_X = {}
    train_data_Y = {}
    for user_id in range(num_users):
        user_X = []
        user_Y = []
        for cell in range(num_memory_cells):
            user_X.append(user_data_memory[user_id][cell]['x'])
            user_Y.append(user_data_memory[user_id][cell]['y'])
        train_data_X[user_id] = np.concatenate(user_X, axis=0)
        train_data_Y[user_id] = np.concatenate(user_Y, axis=0)

    for seed_index, seed in enumerate(seeds_for_avg):
        print(f"************ Seed {seed_count} ************")
        seed_count += 1
        # Define number of classes based on the dataset
        num_classes = 10  # CIFAR-10 has 10 classes

        # Initialize the model
        model = ResNet18(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        w_before_train = [param.data.clone() for param in model.parameters()]

        memory_matrix = [[torch.zeros_like(param).to(device) for param in w_before_train] for _ in range(num_users)]

        for timeframe in range(num_timeframes):
            print(f"******** Timeframe {timeframe + 1} ********")

            # Reset user_new_info_dict when a new phase begins
            if timeframe % phase == 0:
                user_new_info_dict = {user: 0 for user in range(num_users)}

            train_data_X, train_data_Y, user_new_info_dict, current_round_user_data_info = apply_concept_drift(train_data_X, train_data_Y, 
                                                                                                               num_users, x_train, y_train, arrival_rate, 
                                                                                                               timeframe, cell_size, num_memory_cells, user_new_info_dict)

            if timeframe > 0:
                with torch.no_grad():
                    for param, w in zip(model.parameters(), new_weights):
                        param.copy_(w)                
            torch.cuda.empty_cache()

            sparse_gradient = [[torch.zeros_like(param).to(device) for param in w_before_train] for _ in range(num_users)]

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in testloader:
                    # Move images to device; convert labels to NumPy array for mapping
                    images = images.to(device)
                    labels_np = labels.numpy()
        
                    # Get the model outputs and original predictions (10 classes)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
        
                    # Map both predictions and labels to the 5 new classes
                    mapped_preds = torch.tensor(map_to_new_classes(preds.cpu().numpy())).to(device)
                    mapped_labels = torch.tensor(map_to_new_classes(labels_np)).to(device)
        
                    # Update counts: Compare the mapped predictions with mapped labels
                    correct += (mapped_preds == mapped_labels).sum().item()
                    total += labels.size(0)
        
            initial_accuracy = 100 * correct / total

            print(f"Initial Accuracy at Timeframe {timeframe + 1}: {initial_accuracy:.2f}%")

            # ------------- Begin User Training Loop -------------
            user_gradients = []
            for user_id in range(num_users):
                print(f"User: {user_id + 1}")

                # Reset model weights to the initial weights before each user's local training
                with torch.no_grad():
                    for param, saved in zip(model.parameters(), w_before_train):
                        param.copy_(saved)
                torch.cuda.empty_cache()

                # Retrieve the user's training data (combined from all memory cells)
                X_train_u = train_data_X[user_id]
                Y_train_u = train_data_Y[user_id]

                # Shuffle the user's data
                shuffler = np.random.permutation(len(X_train_u))
                X_train_u, Y_train_u = X_train_u[shuffler], Y_train_u[shuffler]

                # Convert numpy arrays to torch tensors and send to device
                X_train_u_tensor = torch.tensor(X_train_u).permute(0, 3, 1, 2).to(device)
                Y_train_u_tensor = torch.tensor(Y_train_u).to(device)

                for epoch in range(epochs):
                    optimizer.zero_grad()
                    # Normalize input; note: original images are assumed to be in [0, 255]
                    X_train_u_tensor = (X_train_u_tensor / 255.0).float()                    
                    loss = criterion(model(X_train_u_tensor), Y_train_u_tensor)
                    loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()

                w_after_train = [param.data.clone() for param in model.parameters()]
                gradient_diff = calculate_gradient_difference(w_before_train, w_after_train)
                gradient_diff_memory = [gradient_diff[j] + memory_matrix[user_id][j] for j in range(len(gradient_diff))]

                if use_memory_matrix:
                    sparse_gradient[user_id] = top_k_sparsificate_model_weights(gradient_diff_memory, fraction[0])
                else:
                    sparse_gradient[user_id] = top_k_sparsificate_model_weights(gradient_diff, fraction[0])

                for j in range(len(w_before_train)):
                    memory_matrix[user_id][j] = (gamma_momentum[0] * memory_matrix[user_id][j]
                                                 + gradient_diff_memory[j] - sparse_gradient[user_id][j])

                gradient_l2_norm = torch.norm(torch.stack([torch.norm(g) for g in gradient_diff])).item()
                gradient_l2_norm_memory = torch.norm(torch.stack([torch.norm(g) for g in gradient_diff_memory])).item()

                if use_memory_matrix:
                    user_gradients.append((user_id, gradient_l2_norm_memory, gradient_diff_memory))
                    loc_grad_mag_memory[run, seed_index, timeframe, user_id] = gradient_l2_norm_memory

                    memory_matrix_norm = sum(torch.norm(param) for param in memory_matrix[user_id])
                    memory_matrix_mag[run, seed_index, timeframe, user_id] = memory_matrix_norm.item()
                else:
                    user_gradients.append((user_id, gradient_l2_norm, gradient_diff))
                    loc_grad_mag[run, seed_index, timeframe, user_id] = gradient_l2_norm

            user_gradients.sort(key=lambda x: x[1], reverse=True)

            # Initialize FL system
            fl_system = FederatedLearning(
            selected_mode, slotted_aloha, num_users, num_slots, sparse_gradient, tx_prob, w_before_train, device, user_new_info_dict, current_round_user_data_info
            )

            # Run the FL mode and get updated weights
            sum_terms, packets_received, num_distinct_users = fl_system.run()

            if num_distinct_users > 0:
                new_weights = [w_before_train[i] + sum_terms[i] / num_distinct_users for i in range(len(w_before_train))]
            else:
                new_weights = [param.clone() for param in w_before_train]
            
            with torch.no_grad():
                for param, w in zip(model.parameters(), new_weights):
                    param.copy_(w)


            per_class_accuracies, accuracy = evaluate_per_class_accuracy(model, testloader, device, num_classes=5)
            
            # Evaluate with additional metrics and save the results
            conf_matrix, f1_across_class, f1_per_class = evaluate_with_metrics(model, testloader, device)

            # Store results and check if this is the best accuracy so far
            accuracy_distributions[run][seed_index][timeframe] = accuracy
            accuracy_distributions_class_0[run][seed_index][timeframe] = per_class_accuracies[0]
            accuracy_distributions_class_1[run][seed_index][timeframe] = per_class_accuracies[1]
            accuracy_distributions_class_2[run][seed_index][timeframe] = per_class_accuracies[2]
            accuracy_distributions_class_3[run][seed_index][timeframe] = per_class_accuracies[3]
            accuracy_distributions_class_4[run][seed_index][timeframe] = per_class_accuracies[4]

            conf_matrix_stats[run][seed_index][timeframe] = conf_matrix
            f1_stats[run][seed_index][timeframe] = f1_across_class
            f1_per_class_stats[run][seed_index][timeframe] = f1_per_class

            # Calculate the update to the weights
            weight_update = [new_weights[i] - w_before_train[i] for i in range(len(w_before_train))]

            # Calculate the L2 norm of the weight update
            update_l2_norm = torch.norm(torch.stack([torch.norm(g) for g in weight_update])).item()

            # Store the global gradient magnitude
            global_grad_mag[run, seed_index, timeframe] = update_l2_norm

            correctly_received_packets_stats[run][seed_index][timeframe]['mean'] = packets_received
            correctly_received_packets_stats[run][seed_index][timeframe]['variance'] = 0

            successful_users_record[run, seed_index, timeframe] = packets_received

            with torch.no_grad():
                for param, w in zip(model.parameters(), new_weights):
                    param.copy_(w)
            w_before_train = new_weights
            torch.cuda.empty_cache()

            print(f"Mean Accuracy at Timeframe {timeframe + 1}: {accuracy:.2f}%")

# Prepare data for saving
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = f"./results10slot1mem_{current_time}"
os.makedirs(save_dir, exist_ok=True)

# Save final results
final_results = []
for run in range(num_runs):
    for seed_index, seed in enumerate(seeds_for_avg):
        for timeframe in range(num_timeframes):
            final_results.append({
                'Run': run,
                'Seed': seed,
                'Timeframe': timeframe + 1,
                'Accuracy': accuracy_distributions[run][seed_index][timeframe],
                'Global Gradient Magnitude': global_grad_mag[run, seed_index, timeframe],
                'Packets Received': correctly_received_packets_stats[run][seed_index][timeframe]['mean'],
                'Variance Packets': correctly_received_packets_stats[run][seed_index][timeframe]['variance']
            })

            # Add additional per-timeframe statistics, independent of num_active_users
            final_results.append({
                'Run': run,
                'Seed': seed,
                'Timeframe': timeframe + 1,
                'Best Global Grad Mag': global_grad_mag[run, seed_index, timeframe],
                'Local Grad Mag': loc_grad_mag[run, seed_index, timeframe].tolist(),
                'Local Grad Mag with Memory': loc_grad_mag_memory[run, seed_index, timeframe].tolist(),
                'Memory Matrix Magnitude': memory_matrix_mag[run, seed_index, timeframe].tolist(),
                'Best Accuracy': accuracy_distributions[run][seed_index][timeframe],
                'Best-Successful Users': successful_users_record[run, seed_index, timeframe]
            })


final_results_df = pd.DataFrame(final_results)
file_path = os.path.join(save_dir, 'final_results.csv')
final_results_df.to_csv(file_path, index=False)
print(f"Final results saved to: {file_path}")

# Save the number of successful users record to CSV
successful_users_record_file_path = os.path.join(save_dir, 'successful_users_record.csv')
# Open the file in write mode
with open(successful_users_record_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Best Packets Received\n')

    # Iterate over runs, seeds, and timeframes to write the best packets received
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                best_packets_received = successful_users_record[run, seed_index, timeframe]
                f.write(f'{run},{seed},{timeframe + 1},{best_packets_received}\n')

print(f"Successful users record saved to: {successful_users_record_file_path}")

loc_grad_mag_file_path = os.path.join(save_dir, 'loc_grad_mag.csv')
# Open the file in write mode
with open(loc_grad_mag_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,User,Local Gradient Magnitude\n')

    # Iterate over runs, seeds, and timeframes to write the local gradient magnitudes
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                # Convert the list of local gradient magnitudes to a string format
                local_gradient_magnitudes = loc_grad_mag[run, seed_index, timeframe]

                # Write each user's local gradient magnitude
                for user_id, grad_mag in enumerate(local_gradient_magnitudes):
                    f.write(f'{run},{seed},{timeframe + 1},{user_id},{grad_mag}\n')

print(f"Local gradient magnitudes saved to: {loc_grad_mag_file_path}")

loc_grad_mag_memory_file_path = os.path.join(save_dir, 'loc_grad_mag_memory.csv')
# Open the file in write mode
with open(loc_grad_mag_memory_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,User,Local Gradient Magnitude\n')

    # Iterate over runs, seeds, and timeframes to write the local gradient magnitudes
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                # Convert the list of local gradient magnitudes to a string format
                local_gradient_magnitudes_memory = loc_grad_mag_memory[run, seed_index, timeframe]

                # Write each user's local gradient magnitude
                for user_id, grad_mag in enumerate(local_gradient_magnitudes_memory):
                    f.write(f'{run},{seed},{timeframe + 1},{user_id},{grad_mag}\n')

print(f"Local gradient magnitudes saved to: {loc_grad_mag_memory_file_path}")

# Save global gradient magnitude
distributions_file_path = os.path.join(save_dir, 'global_grad_mag.csv')
# Open the file in write mode
with open(distributions_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Global Grad Mag\n')

    # Iterate over runs, seeds, and timeframes to write the global gradient magnitudes
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                global_grad_mag_value = global_grad_mag[run, seed_index, timeframe]
                f.write(f'{run},{seed},{timeframe + 1},{global_grad_mag_value}\n')

print(f"Global gradient magnitudes saved to: {distributions_file_path}")

# Save memory matrix magnitudes to CSV
memory_matrix_mag_file_path = os.path.join(save_dir, 'memory_matrix_mag.csv')
# Open the file in write mode
with open(memory_matrix_mag_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,User,Memory Matrix Magnitude\n')

    # Iterate over runs, seeds, and timeframes to write the memory matrix magnitudes
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                # Get the list of memory matrix magnitudes for all users
                memory_magnitudes = memory_matrix_mag[run, seed_index, timeframe].tolist()

                # Iterate over the users to write each user's memory matrix magnitude
                for user_id, memory_magnitude in enumerate(memory_magnitudes):
                    f.write(f'{run},{seed},{timeframe + 1},{user_id},{memory_magnitude}\n')

print(f"Memory matrix magnitudes saved to: {memory_matrix_mag_file_path}")

# Accuracy distribution
distributions_file_path = os.path.join(save_dir, 'accuracy_distributions.csv')
# Open the file in write mode
with open(distributions_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Accuracy\n')
    # Iterate over runs, seeds, and timeframes to write the accuracies
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                accuracy = accuracy_distributions[run][seed_index][timeframe]  # Adjust indexing to exclude num_active_users
                f.write(f'{run},{seed},{timeframe + 1},{accuracy}\n')
print(f"Accuracy distributions saved to: {distributions_file_path}")

# Accuracy distribution (Class 0)
distributions_class_0_file_path = os.path.join(save_dir, 'accuracy_distributions_class_0.csv')
# Open the file in write mode
with open(distributions_class_0_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Accuracy\n')
    # Iterate over runs, seeds, and timeframes to write the accuracies
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                accuracy = accuracy_distributions_class_0[run][seed_index][timeframe]  # Adjust indexing to exclude num_active_users
                f.write(f'{run},{seed},{timeframe + 1},{accuracy}\n')
print(f"Accuracy distributions saved to: {distributions_class_0_file_path}")

# Accuracy distribution (Class 1)
distributions_class_1_file_path = os.path.join(save_dir, 'accuracy_distributions_class_1.csv')
# Open the file in write mode
with open(distributions_class_1_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Accuracy\n')
    # Iterate over runs, seeds, and timeframes to write the accuracies
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                accuracy = accuracy_distributions_class_1[run][seed_index][timeframe]  # Adjust indexing to exclude num_active_users
                f.write(f'{run},{seed},{timeframe + 1},{accuracy}\n')
print(f"Accuracy distributions saved to: {distributions_class_1_file_path}")

# Accuracy distribution (Class 2)
distributions_class_2_file_path = os.path.join(save_dir, 'accuracy_distributions_class_2.csv')
# Open the file in write mode
with open(distributions_class_2_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Accuracy\n')
    # Iterate over runs, seeds, and timeframes to write the accuracies
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                accuracy = accuracy_distributions_class_2[run][seed_index][timeframe]  # Adjust indexing to exclude num_active_users
                f.write(f'{run},{seed},{timeframe + 1},{accuracy}\n')
print(f"Accuracy distributions saved to: {distributions_class_2_file_path}")

# Accuracy distribution (Class 3)
distributions_class_3_file_path = os.path.join(save_dir, 'accuracy_distributions_class_3.csv')
# Open the file in write mode
with open(distributions_class_3_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Accuracy\n')
    # Iterate over runs, seeds, and timeframes to write the accuracies
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                accuracy = accuracy_distributions_class_3[run][seed_index][timeframe]  # Adjust indexing to exclude num_active_users
                f.write(f'{run},{seed},{timeframe + 1},{accuracy}\n')
print(f"Accuracy distributions saved to: {distributions_class_3_file_path}")

# Accuracy distribution (Class 4)
distributions_class_4_file_path = os.path.join(save_dir, 'accuracy_distributions_class_4.csv')
# Open the file in write mode
with open(distributions_class_4_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Accuracy\n')
    # Iterate over runs, seeds, and timeframes to write the accuracies
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                accuracy = accuracy_distributions_class_4[run][seed_index][timeframe]  # Adjust indexing to exclude num_active_users
                f.write(f'{run},{seed},{timeframe + 1},{accuracy}\n')
print(f"Accuracy distributions saved to: {distributions_class_4_file_path}")

# Confusion matrix
conf_matrix_file_path = os.path.join(save_dir, 'confusion_matrix_distributions.csv')
with open(conf_matrix_file_path, 'w') as f:
    # Write CSV header
    f.write('Run,Seed,Timeframe,Confusion_Matrix\n')
    
    for run in conf_matrix_stats:
        for seed_index in conf_matrix_stats[run]:
            for timeframe in conf_matrix_stats[run][seed_index]:
                # When saving the confusion matrix we have to flatten it first, when analyzing remember to restore it
                conf_matrix_flat = np.array(conf_matrix_stats[run][seed_index][timeframe]).flatten()
                conf_matrix_str = ','.join(map(str, conf_matrix_flat))
                f.write(f'{run},{seed_index},{timeframe + 1},{conf_matrix_str}\n')

print(f"Confusion matrices saved to: {conf_matrix_file_path}")

# F1 score
f1_file_path = os.path.join(save_dir, 'f1_score_distributions.csv')
with open(f1_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,F1_Score\n')

    # Write the data rows
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                f1_score_val = f1_stats[run][seed_index][timeframe]
                f.write(f'{run},{seed},{timeframe + 1},{f1_score_val:.4f}\n')
print(f"F1 score distributions saved to: {f1_file_path}")

# F1 score per class
f1_file_path = os.path.join(save_dir, 'f1_score_per_class_distributions.csv')
with open(f1_file_path, 'w') as f:
    # Write the header row
    header = 'Run,Seed,Timeframe,' + ','.join([f'F1_Class_{i}' for i in range(num_classes)]) + '\n'
    f.write(header)

    # Write the data rows
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                f1_score_vals = f1_per_class_stats[run][seed_index][timeframe]

                if isinstance(f1_score_vals, (list, np.ndarray)):
                    f1_str = ','.join([f'{score:.4f}' for score in f1_score_vals])
                else:
                    # fallback: just write a single number
                    f1_str = f'{f1_score_vals:.4f}'

                f.write(f'{run},{seed},{timeframe + 1},{f1_str}\n')
print(f"Per-class F1 score distributions saved to: {f1_file_path}")

# Save correctly received packets statistics to CSV
packets_stats_file_path = os.path.join(save_dir, 'correctly_received_packets_stats.csv')
# Open the file in write mode
with open(packets_stats_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Mean Packets Received,Variance\n')
    # Iterate over runs, seeds, and timeframes to write the packet statistics
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                mean_packets = correctly_received_packets_stats[run][seed_index][timeframe]['mean']  # Adjust indexing
                variance_packets = correctly_received_packets_stats[run][seed_index][timeframe]['variance']  # Adjust indexing
                f.write(f'{run},{seed},{timeframe + 1},{mean_packets},{variance_packets}\n')
print(f"Correctly received packets statistics saved to: {packets_stats_file_path}")

# Save whether each user recieves new data per phase to CSV
new_stale_data_info_file_path = os.path.join(save_dir, 'new_stale_data_info.csv')
# Open the file in write mode
with open(new_stale_data_info_file_path, 'w') as f:
    # Write the header row
    header = "Run,Seed,Timeframe," + ",".join([f"User_{i}" for i in range(num_users)]) + "\n"
    f.write(header)

    # Write the data
    for run in range(num_runs):
        for seed_idx in range(len(seeds_for_avg)):
            for timeframe in range(num_timeframes):  # Iterate directly over timeframes
                # Create the row
                row = f"{run},{seed_idx},{timeframe}"

                # Get the user data for this timeframe (if it exists)
                timeframe_data = new_stale_data_info[run][seed_idx][timeframe]

                # Add user data
                if timeframe_data:  # If we have data for this timeframe
                    for user in range(num_users):
                        value = str(timeframe_data.get(user, 0))  # Store actual drift count
                        row += f",{value}"
                else:  # If no data for this timeframe, fill with zeros
                    row += "," + ",".join(["0"] * num_users)

                f.write(row + "\n")
print(f"User's data status per phase is saved to: {new_stale_data_info_file_path}")

# Record end time and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Save run summary
summary_content = (
    f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n"
    f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n"
    f"Elapsed Time: {elapsed_time:.2f} seconds\n"
    f"Arguments: {vars(args)}\n"
)

summary_file_path = os.path.join(save_dir, 'run_summary.txt')
with open(summary_file_path, 'w') as summary_file:
    summary_file.write(summary_content)

print(f"Run summary saved to: {summary_file_path}")
