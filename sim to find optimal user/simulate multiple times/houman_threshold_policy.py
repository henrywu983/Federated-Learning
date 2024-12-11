import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import pandas as pd
import argparse
import os
from datetime import datetime
import sys
import random as rnd

# Start time
start_time = time.time()

from scipy.io import loadmat

# Path to the .mat file
mat_file_path = 'ThresholdVector.mat'

# Load the .mat file
mat_data = loadmat(mat_file_path)

# Assuming the matrix is stored under a variable name (e.g., 'thresholdvector')
# Replace 'thresholdvector' with the actual variable name in your .mat file
threshold_vector = mat_data['ThresholdVector']

# Convert it to a Python list for easier handling (optional)
threshold_list = threshold_vector.flatten().tolist()

# Print the threshold values
print("Threshold values:", threshold_list)
# Simulate command-line arguments
sys.argv = [
    'placeholder_script_name',
    '--learning_rate', '0.01',
    '--epochs', '3',
    '--batch_size', '64',
    '--num_users', '10',
    '--fraction', '0.2',
    '--transmission_probability', '0.1',
    '--num_slots', '5',
    '--num_timeframes', '15',
    '--seeds', '56','85', '12','29','42',
    '--gamma_momentum', '0.2',
    '--use_memory_matrix', 'true',
    '--threshold'
] + list(map(str, threshold_list))

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
parser.add_argument('--threshold', type=float, nargs='+', help='List of thresholds for each timeframe')
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
use_memory_matrix = args.use_memory_matrix.lower()
threshold_list = args.threshold

# Print out the threshold list to confirm
print(threshold_list)
print(f"use_memory_matrix: {use_memory_matrix}")
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n{'*' * 50}\n*** Using device: {device} ***\n{'*' * 50}\n")

# Transformations for CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Classes in CIFAR-10
classes = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

# VGG16 Model
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features
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

# Prepare data
size_of_user_ds = len(trainset) // num_users
train_data_X = torch.zeros((num_users, size_of_user_ds, 3, 32, 32), device=device)
train_data_Y = torch.ones((num_users, size_of_user_ds), dtype=torch.long, device=device)

for i in range(num_users):
    indices = list(range(size_of_user_ds * i, size_of_user_ds * (i + 1)))
    subset = torch.utils.data.Subset(trainset, indices)
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=size_of_user_ds, shuffle=False)
    for data, target in subset_loader:
        train_data_X[i] = data
        train_data_Y[i] = target

# Additional settings
num_active_users_range = range(1, num_users + 1)

# Initialize matrices for results
num_runs = 5  # only simulate the training process once
num_active_users_record = np.zeros((num_runs, len(seeds_for_avg), num_timeframes))
# Initialize matrices for results with an additional dimension for num_active_users
best_globalgradmag = np.zeros((num_runs, len(seeds_for_avg), num_timeframes))

# Adjust other relevant matrices similarly
successful_users_record = np.zeros((num_runs, len(seeds_for_avg), num_timeframes))
loc_grad_mag = np.zeros((num_runs, len(seeds_for_avg), num_timeframes, num_users, num_users))
loc_grad_mag_memory = np.zeros((num_runs, len(seeds_for_avg), num_timeframes, num_users, num_users))
memory_matrix_mag = np.zeros((num_runs, len(seeds_for_avg), num_timeframes, num_users, num_users))


best_accuracy_distributions = {
    run: {
        seed_index: {timeframe: [] for timeframe in range(num_timeframes)} for seed_index in range(len(seeds_for_avg))
    } for run in range(num_runs)
}
correctly_received_packets_stats = {
    run: {
        seed_index: {
            timeframe: {
                num_active_users: {'mean': None, 'variance': None} for num_active_users in num_active_users_range
            } for timeframe in range(num_timeframes)
        } for seed_index in range(len(seeds_for_avg))
    } for run in range(num_runs)
}
# Main training loop
seed_count = 1

for run in range(num_runs):
    rnd.seed(run)
    np.random.seed(run)
    torch.manual_seed(run)
    print(f"************ Run {run + 1} ************")
    for seed_index, seed in enumerate(seeds_for_avg):
        print(f"************ Seed {seed_count} ************")
        seed_count += 1
        # Define number of classes based on the dataset
        num_classes = 10  # CIFAR-10 has 10 classes

        # Initialize the model
        model = VGG16(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        w_before_train = [param.data.clone() for param in model.parameters()]

        for timeframe in range(num_timeframes):
            print(f"******** Timeframe {timeframe + 1} with Threshold {threshold_list[timeframe]}********")
            if timeframe > 0:
                model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), new_weights)})
            torch.cuda.empty_cache()

            memory_matrix = [[torch.zeros_like(param).to(device) for param in w_before_train] for _ in range(num_users)]
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
                    loss = criterion(model(X_train_u), Y_train_u)
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
            current_threshold = threshold_list[timeframe]
            active_users = [user_id for user_id, grad_norm, _ in user_gradients if grad_norm > current_threshold]

            if active_users:
                tx_prob = 1 / len(active_users)
            else:
                while not active_users:
                    current_threshold *= 0.99
                    active_users = [user_id for user_id, grad_norm, _ in user_gradients if grad_norm > current_threshold]
                tx_prob = 1 / len(active_users)
            num_active_users = len(active_users)
            print(f"*** {num_active_users} Active User(s) ***")

            sum_terms = [torch.zeros_like(param).to(device) for param in w_before_train]
            packets_received = 0
            distinct_users = set()

            while packets_received == 0:  # Retry until packets_received is not zero
                for _ in range(num_slots):
                    successful_users = simulate_transmissions(num_active_users, tx_prob)
                    if successful_users:
                        success_user = active_users[successful_users[0]]
                        if success_user not in distinct_users:
                            sum_terms = [sum_terms[j] + sparse_gradient[success_user][j] for j in range(len(sum_terms))]
                            packets_received += 1
                            distinct_users.add(success_user)


            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users}")
            new_weights = [w_before_train[i] + sum_terms[i] / num_distinct_users for i in range(len(w_before_train))] if num_distinct_users > 0 else [param.clone() for param in w_before_train]

            model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), new_weights)})

            with torch.no_grad():
                correct = sum((model(images.to(device)).argmax(dim=1) == labels.to(device)).sum().item()
                              for images, labels in testloader)
            accuracy = 100 * correct / len(testset)
            print(f"Accuracy with {num_active_users} active users: {accuracy:.2f}%")

            # Calculate the update to the weights
            weight_update = [new_weights[i] - w_before_train[i] for i in range(len(w_before_train))]

            # Calculate the L2 norm of the weight update
            update_l2_norm = torch.norm(torch.stack([torch.norm(g) for g in weight_update])).item()

            # Store the global gradient magnitude
            best_accuracy_distributions[run][seed_index][timeframe] = accuracy
            num_active_users_record[run, seed_index, timeframe] = num_active_users
            successful_users_record[run, seed_index, timeframe] = num_distinct_users
            best_globalgradmag[run,seed_index,timeframe] = update_l2_norm

            model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), new_weights)})
            w_before_train = new_weights
            torch.cuda.empty_cache()

            print(f"Number of active users: {num_active_users}")
            print(f"Mean Accuracy at Timeframe {timeframe + 1}: {accuracy:.2f}%")

# Prepare data for saving
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = f"./thresholdedresults10slot1mem_{current_time}"
os.makedirs(save_dir, exist_ok=True)

# Save final results
final_results = []
for run in range(num_runs):
    for seed_index, seed in enumerate(seeds_for_avg):
        for timeframe in range(num_timeframes):
                        # Additionally store the 'Successful Users' data, which is independent of num_active_users
            final_results.append({
                'Run': run,
                'Seed': seed,
                'Timeframe': timeframe + 1,
                'Num Active Users': 'N/A',  # Use 'N/A' or similar to indicate this row is not tied to num_active_users
                'Best Global Grad Mag': best_globalgradmag[run,seed_index,timeframe],  # Or leave this field out entirely
                'Local Grad Mag': loc_grad_mag[run, seed_index, timeframe].tolist(),   # Or leave this field out entirely
                'Local Grad Mag with Memory': loc_grad_mag_memory[run, seed_index, timeframe].tolist(),
                'Memory Matrix Magnitude': memory_matrix_mag[run, seed_index, timeframe].tolist(),
                'Best Accuracy': best_accuracy_distributions[run][seed_index][timeframe],  # Store the best accuracy
                'Best-Successful Users': successful_users_record[run, seed_index, timeframe],
                'BestActiveUsers': num_active_users_record[run, seed_index, timeframe]
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

num_active_users_record_file_path = os.path.join(save_dir, 'num_active_users_record.csv')

# Open the file in write mode
with open(num_active_users_record_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Best Num Active Users\n')
    
    # Iterate over runs, seeds, and timeframes to write the best number of active users
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                best_num_active_users = num_active_users_record[run, seed_index, timeframe]
                f.write(f'{run},{seed},{timeframe + 1},{best_num_active_users}\n')

print(f"Number of active users record saved to: {num_active_users_record_file_path}")


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

# Save global gradient magnitudes to CSV
distributions_file_path = os.path.join(save_dir, 'best_global_grad_mag.csv')
# Open the file in write mode
with open(distributions_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Best Global Grad Mag\n')
    
    # Iterate over runs, seeds, and timeframes to write the best global gradient magnitudes
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                best_global_grad_mag_value = best_globalgradmag[run, seed_index, timeframe]
                f.write(f'{run},{seed},{timeframe + 1},{best_global_grad_mag_value}\n')

print(f"Best global gradient magnitudes saved to: {distributions_file_path}")


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



# Save best accuracy distributions to CSV
distributions_file_path = os.path.join(save_dir, 'best_accuracy_distributions.csv')

# Open the file in write mode
with open(distributions_file_path, 'w') as f:
    # Write the header row
    f.write('Run,Seed,Timeframe,Best Accuracy\n')    
    # Iterate over runs, seeds, and timeframes to write the best accuracies
    for run in range(num_runs):
        for seed_index, seed in enumerate(seeds_for_avg):
            for timeframe in range(num_timeframes):
                best_accuracy = best_accuracy_distributions[run][seed_index][timeframe]
                f.write(f'{run},{seed},{timeframe + 1},{best_accuracy}\n')
print(f"Best accuracy distributions saved to: {distributions_file_path}")
# Save correctly received packets statistics to CSV
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
