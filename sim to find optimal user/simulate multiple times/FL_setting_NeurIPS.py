import random
import numpy as np
import torch
import torch.nn.functional as F

class FederatedLearning:
    def __init__(self, mode, slotted_aloha, num_users, num_slots, sparse_gradient, tx_prob, w_before_train, device, 
                 user_new_info_dict, current_round_user_data_info, prev_round_global_grad, grad_per_user, cos_similarity, 
                 user_accuracies, user_accuracies_increment):
        self.mode = mode
        self.slotted_aloha = slotted_aloha
        self.num_users = num_users
        self.num_slots = num_slots
        self.sparse_gradient = sparse_gradient
        self.tx_prob = tx_prob
        self.w_before_train = w_before_train
        self.device = device
        self.user_new_info_dict = user_new_info_dict
        self.current_round_user_data_info = current_round_user_data_info
        self.prev_round_global_grad = prev_round_global_grad
        self.grad_per_user = grad_per_user
        self.cos_similarity = cos_similarity
        self.user_accuracies = user_accuracies
        self.user_accuracies_increment = user_accuracies_increment

    def lp_cosine_similarity(self, x: torch.Tensor, y: torch.Tensor, p: int = 2) -> float:
        """
        Compute the Lp cosine similarity between two flattened gradient vectors.
    
        Args:
            x (torch.Tensor): 1D tensor.
            y (torch.Tensor): 1D tensor.
            p (int): Norm degree (e.g., 2 for L2).
    
        Returns:
            float: The Lp cosine similarity.
        """
        norm_x = torch.norm(x, p=p)
        norm_y = torch.norm(y, p=p)
        norm_x_plus_y_sq = torch.norm(x + y, p=p) ** 2
        norm_x_sq = norm_x ** 2
        norm_y_sq = norm_y ** 2

        numerator = 0.5 * (norm_x_plus_y_sq - norm_x_sq - norm_y_sq)
        denominator = norm_x * norm_y + 1e-12  # avoid division by zero

        return (numerator / denominator).item()
    
    def simulate_fl_round_genie_aided(self):
        """Handles both Slotted ALOHA and standard user processing."""
        sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_before_train]
        packets_received = 0
        distinct_users = set()

        if self.slotted_aloha == 'true':
            for _ in range(self.num_slots):
                successful_users = self.simulate_transmissions()
                if successful_users:
                    success_user = successful_users[0]
                    if success_user not in distinct_users:
                        sum_terms = [sum_terms[j] + self.sparse_gradient[success_user][j] for j in range(len(sum_terms))]
                        packets_received += 1
                        distinct_users.add(success_user)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users}")
        
        else:
            if self.num_users < 3:
                raise ValueError("Number of users must be at least 3 to ensure proper selection.")

            # Old genie-aided: Sort users by the amount of new data (highest first) and pick the top 3            
            sorted_users = sorted(self.user_new_info_dict.keys(), key=lambda u: self.user_new_info_dict[u], reverse=True)
            selected_users = sorted_users[:3]

            for user_id in selected_users:
                sum_terms = [sum_terms[j] + self.sparse_gradient[user_id][j] for j in range(len(sum_terms))]
                packets_received += 1
                distinct_users.add(user_id)            
            
            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users} (No Slotted ALOHA)")

        return sum_terms, packets_received, num_distinct_users

    def simulate_fl_round_vanilla(self):
        """Handles both Slotted ALOHA and standard user processing."""
        sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_before_train]
        packets_received = 0
        distinct_users = set()

        if self.slotted_aloha == 'true':
            for _ in range(self.num_slots):
                successful_users = self.simulate_transmissions()
                if successful_users:
                    success_user = successful_users[0]
                    if success_user not in distinct_users:
                        sum_terms = [sum_terms[j] + self.sparse_gradient[success_user][j] for j in range(len(sum_terms))]
                        packets_received += 1
                        distinct_users.add(success_user)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users}")
        
        else:
            if self.num_users < 3:
                raise ValueError("Number of users must be at least 3 to ensure proper selection.")
        
            # Select 3 random users to successfully transmit
            selected_users = random.sample(range(self.num_users), 3)

            for user_id in selected_users:
                sum_terms = [sum_terms[j] + self.sparse_gradient[user_id][j] for j in range(len(sum_terms))]
                packets_received += 1
                distinct_users.add(user_id)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users} (No Slotted ALOHA)")

        return sum_terms, packets_received, num_distinct_users
    
    # Compute cos similarities and select top-3 users
    def simulate_fl_round_user_selection_cos(self):
        """Handles both Slotted ALOHA and standard user processing."""
        sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_before_train]
        packets_received = 0
        distinct_users = set()

        if self.slotted_aloha == 'true':
            for _ in range(self.num_slots):
                successful_users = self.simulate_transmissions()
                if successful_users:
                    success_user = successful_users[0]
                    if success_user not in distinct_users:
                        sum_terms = [sum_terms[j] + self.sparse_gradient[success_user][j] for j in range(len(sum_terms))]
                        packets_received += 1
                        distinct_users.add(success_user)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users}")
        
        else:
            cos_dict = {}
            for user_id in range(self.num_users):
                # Flatten gradients into 1D vectors
                user_grad_vector = torch.cat([g.view(-1) for g in self.grad_per_user[user_id]])
                global_grad_vector = torch.cat([g.view(-1) for g in self.prev_round_global_grad])

                # Compute cosine similarity
                lp_cos_val = self.lp_cosine_similarity(user_grad_vector, global_grad_vector, p = self.cos_similarity)
                cos_dict[user_id] = lp_cos_val

            # --- Sort by cos value in descending order ---
            sorted_users = sorted(cos_dict, key=cos_dict.get, reverse=True)
            selected_users = sorted_users[:3]

            for user_id in selected_users:
                sum_terms = [sum_terms[j] + self.sparse_gradient[user_id][j] for j in range(len(sum_terms))]
                packets_received += 1
                distinct_users.add(user_id)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users} (No Slotted ALOHA)")

        return sum_terms, packets_received, num_distinct_users

# Compute cos DIS-similarities and select top-3 users with least values
    def simulate_fl_round_user_selection_cos_dis(self):
        """Handles both Slotted ALOHA and standard user processing."""
        sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_before_train]
        packets_received = 0
        distinct_users = set()

        if self.slotted_aloha == 'true':
            for _ in range(self.num_slots):
                successful_users = self.simulate_transmissions()
                if successful_users:
                    success_user = successful_users[0]
                    if success_user not in distinct_users:
                        sum_terms = [sum_terms[j] + self.sparse_gradient[success_user][j] for j in range(len(sum_terms))]
                        packets_received += 1
                        distinct_users.add(success_user)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users}")
        
        else:
            cos_dict = {}
            for user_id in range(self.num_users):
                # Flatten gradients into 1D vectors
                user_grad_vector = torch.cat([g.view(-1) for g in self.grad_per_user[user_id]])
                global_grad_vector = torch.cat([g.view(-1) for g in self.prev_round_global_grad])

                # Compute cosine similarity
                lp_cos_val = self.lp_cosine_similarity(user_grad_vector, global_grad_vector, p = self.cos_similarity)
                cos_dict[user_id] = lp_cos_val

            # --- Sort by cos value in ascending order ---
            sorted_users = sorted(cos_dict, key=cos_dict.get, reverse=False)
            selected_users = sorted_users[:3]

            for user_id in selected_users:
                sum_terms = [sum_terms[j] + self.sparse_gradient[user_id][j] for j in range(len(sum_terms))]
                packets_received += 1
                distinct_users.add(user_id)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users} (No Slotted ALOHA)")

        return sum_terms, packets_received, num_distinct_users

    # Compare the accuracies from all local users and select top-3 users
    def simulate_fl_round_user_selection_acc(self):
        """Handles both Slotted ALOHA and standard user processing."""
        sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_before_train]
        packets_received = 0
        distinct_users = set()

        if self.slotted_aloha == 'true':
            for _ in range(self.num_slots):
                successful_users = self.simulate_transmissions()
                if successful_users:
                    success_user = successful_users[0]
                    if success_user not in distinct_users:
                        sum_terms = [sum_terms[j] + self.sparse_gradient[success_user][j] for j in range(len(sum_terms))]
                        packets_received += 1
                        distinct_users.add(success_user)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users}")
        
        else:
            user_ids = list(range(self.num_users))
            acc_dict = dict(zip(user_ids, self.user_accuracies[0]))

            # --- Sort by accuracy value in descending order ---
            sorted_users = sorted(acc_dict, key=acc_dict.get, reverse=True)
            selected_users = sorted_users[:3]

            for user_id in selected_users:
                sum_terms = [sum_terms[j] + self.sparse_gradient[user_id][j] for j in range(len(sum_terms))]
                packets_received += 1
                distinct_users.add(user_id)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users} (No Slotted ALOHA)")

        return sum_terms, packets_received, num_distinct_users

    # Compare the accuracies increments from all local users and select top-3 users
    def simulate_fl_round_user_selection_acc_increment(self):
        """Handles both Slotted ALOHA and standard user processing."""
        sum_terms = [torch.zeros_like(param).to(self.device) for param in self.w_before_train]
        packets_received = 0
        distinct_users = set()

        if self.slotted_aloha == 'true':
            for _ in range(self.num_slots):
                successful_users = self.simulate_transmissions()
                if successful_users:
                    success_user = successful_users[0]
                    if success_user not in distinct_users:
                        sum_terms = [sum_terms[j] + self.sparse_gradient[success_user][j] for j in range(len(sum_terms))]
                        packets_received += 1
                        distinct_users.add(success_user)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users}")
        
        else:
            user_ids = list(range(self.num_users))
            acc_dict = dict(zip(user_ids, self.user_accuracies_increment[0]))

            # --- Sort by accuracy value in descending order ---
            sorted_users = sorted(acc_dict, key=acc_dict.get, reverse=True)            
            selected_users = sorted_users[:3]

            for user_id in selected_users:
                sum_terms = [sum_terms[j] + self.sparse_gradient[user_id][j] for j in range(len(sum_terms))]
                packets_received += 1
                distinct_users.add(user_id)

            num_distinct_users = len(distinct_users)
            print(f"Number of distinct clients: {num_distinct_users} (No Slotted ALOHA)")

        return sum_terms, packets_received, num_distinct_users




    def simulate_transmissions(self):
        """Simulates slotted ALOHA transmissions."""
        decisions = np.random.rand(self.num_users) < self.tx_prob
        if np.sum(decisions) == 1:
            return [i for i, decision in enumerate(decisions) if decision]
        return []

    def run(self):
        """Dispatch based on the FL mode."""
        if self.mode == "genie_aided":
            return self.genie_aided()
        elif self.mode == "vanilla":
            return self.vanilla()
        elif self.mode == "user_selection_cos":
            return self.user_selection_cos()
        elif self.mode == "user_selection_acc":
            return self.user_selection_acc()
        elif self.mode == 'user_selection_acc_increment':
            return self.user_selection_acc_increment()
        elif self.mode == 'user_selection_cos_dis':
            return self.user_selection_cos_dis()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def genie_aided(self):
        print("Running Genie-Aided FL...")
        return self.simulate_fl_round_genie_aided()

    def vanilla(self):
        print("Running Vanilla FL...")
        return self.simulate_fl_round_vanilla()

    def user_selection_cos(self):
        print("Running User Selection FL (Cosine similarity)...")
        return self.simulate_fl_round_user_selection_cos()
    
    def user_selection_acc(self):
        print("Running User Selection FL (Accuracy selection)...")
        return self.simulate_fl_round_user_selection_acc()
    
    def user_selection_acc_increment(self):
        print("Running User Selection FL (Accuracy increment selection)...")
        return self.simulate_fl_round_user_selection_acc_increment()
    
    def user_selection_cos_dis(self):
        print("Running User Selection FL (Cosine DIS-similarity)...")
        return self.simulate_fl_round_user_selection_cos_dis()
