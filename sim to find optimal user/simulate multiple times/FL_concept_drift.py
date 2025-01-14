import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import pandas as pd
import argparse
import os
from datetime import datetime
import sys
import random as rnd
import matplotlib.pyplot as plt
# writer = SummaryWriter("runs/fl_concept_drift_adam_00005")
# Start time
start_time = time.time()

# Simulate command-line arguments
sys.argv = [
    'placeholder_script_name',
    '--learning_rate', '0.001',
    '--epochs', '3',
    '--batch_size', '64',
    '--num_users', '10',
    '--fraction', '0.1',
    '--transmission_probability', '0.1',
    '--num_slots', '10',
    '--num_timeframes', '50',
    '--user_data_size', '2500',
    '--seeds', '56', '3', '29', '85', '65',
    '--gamma_momentum', '0',
    '--use_memory_matrix', 'false',
    '--arrival_rate', '0.5',
    '--phase', '5',
    '--num_runs', '5',
    '--slotted_aloha', 'false'
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
parser.add_argument('--phase', type=int, default=5,help='When concept drift happens')
parser.add_argument('--num_runs', type=int, default=5,help='Number of simulations')
parser.add_argument('--slotted_aloha', type=str, default='true',help='Whether we use Slotted aloha in the simulation')

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

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n{'*' * 50}\n*** Using device: {device} ***\n{'*' * 50}\n")

class_0_labels = [0, 1, 2, 3, 4]  # Animals
class_1_labels = [5, 6, 7, 8, 9]  # Objects

# CIFAR-10 dataset and preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
x_train, y_train = trainset.data, np.array(trainset.targets)

assert len(x_train) >= num_users * user_data_size, "Dataset too small for requested user allocation!"

# Classes in CIFAR-10
classes = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

# VGG16 Model
class VGG16(nn.Module):
    def __init__(self, num_classes=10, train_dense_only=False):
        super(VGG16, self).__init__()
        self.features = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features

        # Freeze the feature extractor if train_dense_only is True
        self.train_dense_only = train_dense_only
        if self.train_dense_only:
            for param in self.features.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def print_gpu_memory(msg):
    print(f"{msg}: {torch.cuda.memory_allocated() / 1e6:.2f} MB allocated, {torch.cuda.memory_reserved() / 1e6:.2f} MB reserved")

# Sparsify the model weights
def top_k_sparsificate_model_weights(weights, fraction):
    flat_weights = torch.cat([w.view(-1) for w in weights])
    threshold_value = torch.quantile(torch.abs(flat_weights), 1 - fraction)
    new_weights = []
    for w in weights:
        mask = torch.abs(w) >= threshold_value
        new_weights.append(w * mask.float())
    return new_weights

'''
# Simulate transmissions on the slotted ALOHA channel
def simulate_transmissions(num_users, user_new_info_dict, tx_prob):
    """
    Simulate slotted Aloha transmissions where each user transmits based on their respective transmission probability.
    Users with user_info_dict value True will use tx_prob_new, others will use tx_prob_old.
    """
    # Initialize transmission probabilities
    tx_prob_new = tx_prob
    tx_prob_old = 0

    # Generate transmission decisions for each user based on their respective probabilities
    decisions = np.array([np.random.rand() < (tx_prob_new if user_new_info_dict[user_id] else tx_prob_old) for user_id in range(num_users)])

    # Check if there is only one successful transmission
    if np.sum(decisions) == 1:
        return [i for i, decision in enumerate(decisions) if decision]
    return []
'''

# Simulate transmissions on the slotted ALOHA channel
def simulate_transmissions(num_users, transmission_probability):
    decisions = np.random.rand(num_users) < transmission_probability
    if np.sum(decisions) == 1:
        return [i for i, decision in enumerate(decisions) if decision]
    return []

# Calculate gradient difference between two sets of weights
def calculate_gradient_difference(w_before, w_after):
    return [w_after[k] - w_before[k] for k in range(len(w_after))]

def apply_concept_drift(train_data_X, train_data_Y, class_0_labels, class_1_labels, num_users, x_train, y_train, arrival_rate, timeframe):
    """
    Apply concept drift by discarding 40% of data (from both classes)
    and refilling it with a majority of class 1 data.
    """
    if ((timeframe + 1) // phase) % 2 == 1:
        print("Applying Concept Drift: Discarding 40% of data from both classes and refilling with majority class 1 data.")
    else:
        print("Applying Concept Drift: Discarding 40% of data from both classes and refilling with majority class 0 data.")

    # Record which user has new info.
    user_new_info_dict = {}

    for user_id in range(num_users):
        # Generate a random number to decide whether this user will discard data and refill
        random_value = np.random.rand()
        if random_value < arrival_rate:
            user_new_info_dict[user_id] = True
            # Calculate the number of data points to discard (40% of all data)
            discard_size = int(0.4 * len(train_data_Y[user_id]))

            # Randomly select indices to discard
            all_indices = np.arange(len(train_data_Y[user_id]))
            discard_indices = np.random.choice(all_indices, discard_size, replace=False)

            # Keep only the data not being discarded
            keep_indices = [i for i in range(len(train_data_Y[user_id])) if i not in discard_indices]
            train_data_X[user_id] = train_data_X[user_id][keep_indices]
            train_data_Y[user_id] = train_data_Y[user_id][keep_indices]

            # Redistribute 40% of data from both classes with a majority of class 1
            num_refill_samples = int(0.4 * len(train_data_Y[user_id]))  # Total amount to refill for each user
            if ((timeframe + 1) // phase) % 2 == 1:
                num_class_1_samples = int(0.9 * num_refill_samples)  # 90% from class 1
                num_class_0_samples = num_refill_samples - num_class_1_samples  # Remaining 10% from class 0
            else:
                num_class_0_samples = int(0.9 * num_refill_samples)  # 90% from class 0
                num_class_1_samples = num_refill_samples - num_class_0_samples  # Remaining 10% from class 1

            # Prepare indices for refilling
            class_0_indices = np.where(np.isin(y_train, class_0_labels))[0]
            class_1_indices = np.where(np.isin(y_train, class_1_labels))[0]
            np.random.shuffle(class_0_indices)
            np.random.shuffle(class_1_indices)

            # Select new data from class 0 and class 1 for refilling
            user_class_0_indices = np.random.choice(class_0_indices, num_class_0_samples, replace=False)
            user_class_1_indices = np.random.choice(class_1_indices, num_class_1_samples, replace=False)

            # Combine the new data from both classes
            refill_indices = np.concatenate((user_class_0_indices, user_class_1_indices))
            new_data_X = torch.tensor(x_train[refill_indices]).permute(0, 3, 1, 2).to(device)
            new_data_Y = torch.tensor(y_train[refill_indices]).to(device)

            # Append new data to the user's dataset
            train_data_X[user_id] = torch.cat([train_data_X[user_id], new_data_X], dim=0)
            train_data_Y[user_id] = torch.cat([train_data_Y[user_id], new_data_Y], dim=0)
        else:
            user_new_info_dict[user_id] = False

    return train_data_X, train_data_Y, user_new_info_dict


def plot_user_data_distribution(train_data_Y, num_users, timeframe):
  """
  Plot the label distribution for each user's dataset.

  Args:
      train_data_Y (list): A list of tensors containing labels for each user's data.
       num_users (int): Number of users.
       timeframe (int): Current timeframe for labeling the plot.
   """
  fig, axes = plt.subplots(num_users, 1, figsize=(10, num_users * 2), sharex=True)
  fig.suptitle(f"User Data Distribution at Timeframe {timeframe + 1}", fontsize=16)

  for user_id in range(num_users):
      # Count the number of occurrences for each label
      labels, counts = torch.unique(train_data_Y[user_id], return_counts=True)
      label_counts = dict(zip(labels.cpu().numpy(), counts.cpu().numpy()))

      # Plot the distribution
      axes[user_id].bar(label_counts.keys(), label_counts.values())
      axes[user_id].set_title(f"User {user_id + 1}")
      axes[user_id].set_ylabel("Count")
      axes[user_id].set_xticks(range(10))
      axes[user_id].set_xticklabels([str(i) for i in range(10)])

  # Add x-label to the last plot
  axes[-1].set_xlabel("Labels")
  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.show()

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

# Main training loop
seed_count = 1

for run in range(num_runs):
    rnd.seed(run)
    np.random.seed(run)
    torch.manual_seed(run)
    print(f"************ Run {run + 1} ************")

    # Prepare data
    train_data_X = [[] for _ in range(num_users)]
    train_data_Y = [[] for _ in range(num_users)]

    size_of_user_ds = len(trainset) // num_users
    # Prepare data for users with 70% Class 0 and 30% Class 1
    for i in range(num_users):
        # Select indices for Class 0 and Class 1
        class_0_indices = np.where(np.isin(y_train, class_0_labels))[0]
        class_1_indices = np.where(np.isin(y_train, class_1_labels))[0]

        # Shuffle indices
        np.random.shuffle(class_0_indices)
        np.random.shuffle(class_1_indices)

        # Allocate 70% from Class 0 and 30% from Class 1
        num_class_0_samples = int(0.7 * user_data_size)
        num_class_1_samples = user_data_size - num_class_0_samples

        user_class_0_indices = np.random.choice(class_0_indices, num_class_0_samples, replace=False)
        user_class_1_indices = np.random.choice(class_1_indices, num_class_1_samples, replace=False)

        # Assign data to the user
        train_data_X[i] = torch.tensor(x_train[np.concatenate((user_class_0_indices, user_class_1_indices))]).permute(0, 3, 1, 2).to(device)
        train_data_Y[i] = torch.tensor(y_train[np.concatenate((user_class_0_indices, user_class_1_indices))]).to(device)

    for seed_index, seed in enumerate(seeds_for_avg):
        print(f"************ Seed {seed_count} ************")
        seed_count += 1
        # Define number of classes based on the dataset
        num_classes = 10  # CIFAR-10 has 10 classes

        # Initialize the model
        model = VGG16(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # user_losses = {user_id: [] for user_id in range(num_users)}        
        # global_loss = []

        w_before_train = [param.data.clone() for param in model.parameters()]

        memory_matrix = [[torch.zeros_like(param).to(device) for param in w_before_train] for _ in range(num_users)]

        for timeframe in range(num_timeframes):
            print(f"******** Timeframe {timeframe + 1} ********")

            # Apply Concept Drift at Timeframe 8
            if (timeframe + 1) % phase == 0:
                train_data_X, train_data_Y, user_new_info_dict = apply_concept_drift(train_data_X, train_data_Y, class_0_labels, class_1_labels, num_users, x_train, y_train, arrival_rate, timeframe)

            # plot_user_data_distribution(train_data_Y, num_users, timeframe)

            if timeframe > 0:
                model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), new_weights)})
            torch.cuda.empty_cache()

            sparse_gradient = [[torch.zeros_like(param).to(device) for param in w_before_train] for _ in range(num_users)]

            model.eval()
            with torch.no_grad():
                correct = sum((model(images.to(device)).argmax(dim=1) == labels.to(device)).sum().item()
                              for images, labels in testloader)
            initial_accuracy = 100 * correct / len(testset)
            print(f"Initial Accuracy at Timeframe {timeframe + 1}: {initial_accuracy:.2f}%")

            user_gradients = []
            for user_id in range(num_users):
                print(f"User: {user_id + 1}")
                model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), w_before_train)})
                torch.cuda.empty_cache()

                X_train_u, Y_train_u = train_data_X[user_id], train_data_Y[user_id]
                shuffler = np.random.permutation(len(X_train_u))
                X_train_u, Y_train_u = X_train_u[shuffler], Y_train_u[shuffler]

                for epoch in range(epochs):
                    optimizer.zero_grad()
                    X_train_u = (X_train_u / 255.0).float()
                    loss = criterion(model(X_train_u), Y_train_u)
                    #print_gpu_memory("Before training")
                    loss.backward()
                    #print_gpu_memory("After backward pass")
                    optimizer.step()
                    #print_gpu_memory("After optimizer step")
                    torch.cuda.empty_cache()

                    # Log loss
                    # user_losses[user_id].append(loss.item())

                # Log loss and accuracy for TensorBoard
                # writer.add_scalar(f"User_{user_id+1}/Loss", user_losses[user_id][-1], timeframe)                
                
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

            sum_terms = [torch.zeros_like(param).to(device) for param in w_before_train]
            packets_received = 0
            distinct_users = set()

            if slotted_aloha == 'true':
                for _ in range(num_slots):
                    successful_users = simulate_transmissions(num_users, tx_prob)
                    if successful_users:
                        success_user = successful_users[0]
                        if success_user not in distinct_users:
                            sum_terms = [sum_terms[j] + sparse_gradient[success_user][j] for j in range(len(sum_terms))]
                            packets_received += 1
                            distinct_users.add(success_user)

                num_distinct_users = len(distinct_users)
                print(f"Number of distinct clients: {num_distinct_users}")
            else:
                for user_id in range(num_users):
                    sum_terms = [sum_terms[j] + sparse_gradient[user_id][j] for j in range(len(sum_terms))]
                    packets_received += 1

                num_distinct_users = num_users
                print(f"Number of distinct clients: {num_distinct_users} (No Slotted Aloha)")

            if num_distinct_users > 0:
                new_weights = [w_before_train[i] + sum_terms[i] / num_distinct_users for i in range(len(w_before_train))]
            else:
                new_weights = [param.clone() for param in w_before_train]

            model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), new_weights)})

            with torch.no_grad():
                correct_class_0 = 0
                correct_class_1 = 0
                total_class_0 = 0
                total_class_1 = 0

                for images, labels in testloader:
                    outputs = model(images.to(device))
                    predictions = outputs.argmax(dim=1)
                    labels = labels.to(device)

                    # Count correct predictions for Class 0
                    for i in range(len(labels)):
                        if labels[i].item() in class_0_labels:
                            total_class_0 += 1
                            if predictions[i] == labels[i]:
                                correct_class_0 += 1

                    # Count correct predictions for Class 1
                    for i in range(len(labels)):
                        if labels[i].item() in class_1_labels:
                            total_class_1 += 1
                            if predictions[i] == labels[i]:
                                correct_class_1 += 1
                    
                # Calculate accuracy for each class
                accuracy_class_0 = 100 * correct_class_0 / total_class_0 if total_class_0 > 0 else 0
                accuracy_class_1 = 100 * correct_class_1 / total_class_1 if total_class_1 > 0 else 0

                print(f"Accuracy for Class 0 (Animals): {accuracy_class_0:.2f}%")
                print(f"Accuracy for Class 1 (Objects): {accuracy_class_1:.2f}%")

                correct = sum((model(images.to(device)).argmax(dim=1) == labels.to(device)).sum().item()
                                  for images, labels in testloader)
            
            # Evaluate global model loss
            # global_loss_sum = 0
            # with torch.no_grad():
            #     for images, labels in testloader:
            #         outputs = model(images.to(device))
            #         loss = criterion(outputs, labels.to(device))
            #         global_loss_sum += loss.item()
            # global_loss.append(global_loss_sum / len(testloader))
            # writer.add_scalar("Global/Loss", global_loss[-1], timeframe)

            accuracy = 100 * correct / len(testset)

            # Log global accuracy to TensorBoard
            # writer.add_scalar("Global/Accuracy", accuracy, timeframe)

            # Store results and check if this is the best accuracy so far
            accuracy_distributions[run][seed_index][timeframe] = accuracy
            accuracy_distributions_class_0[run][seed_index][timeframe] = accuracy_class_0
            accuracy_distributions_class_1[run][seed_index][timeframe] = accuracy_class_1

            # Calculate the update to the weights
            weight_update = [new_weights[i] - w_before_train[i] for i in range(len(w_before_train))]

            # Calculate the L2 norm of the weight update
            update_l2_norm = torch.norm(torch.stack([torch.norm(g) for g in weight_update])).item()

            # Store the global gradient magnitude
            global_grad_mag[run, seed_index, timeframe] = update_l2_norm

            correctly_received_packets_stats[run][seed_index][timeframe]['mean'] = packets_received
            correctly_received_packets_stats[run][seed_index][timeframe]['variance'] = 0

            successful_users_record[run, seed_index, timeframe] = packets_received

            model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), new_weights)})
            w_before_train = new_weights
            torch.cuda.empty_cache()

            print(f"Mean Accuracy at Timeframe {timeframe + 1}: {accuracy:.2f}%")

        # Close the TensorBoard writer
        # writer.close()

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
