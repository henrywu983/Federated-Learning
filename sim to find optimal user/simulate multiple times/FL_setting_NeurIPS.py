import random
import numpy as np
import torch

class FederatedLearning:
    def __init__(self, mode, slotted_aloha, num_users, num_slots, sparse_gradient, tx_prob, w_before_train, device, user_new_info_dict, current_round_user_data_info):
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
    
    # NOT DONE
    def simulate_fl_round_user_selection(self):
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
            for user_id in range(self.num_users):
                sum_terms = [sum_terms[j] + self.sparse_gradient[user_id][j] for j in range(len(sum_terms))]
                packets_received += 1

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
        elif self.mode == "user_selection":
            return self.user_selection()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def genie_aided(self):
        print("Running Genie-Aided FL...")
        return self.simulate_fl_round_genie_aided()

    def vanilla(self):
        print("Running Vanilla FL...")
        return self.simulate_fl_round_vanilla()

    def user_selection(self):
        print("Running User Selection FL...")
        return self.simulate_fl_round_user_selection()
