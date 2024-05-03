import gym
from gym import spaces
import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os


def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points 
        on the earth (specified in decimal degrees).
        """
        # Convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # Haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. 
        return c * r

class DeliveryRouteEnv(gym.Env):
    """Custom Environment for optimizing delivery routes using DQN."""
    metadata = {'render.modes': ['console']}

    def __init__(self, data, phase='train', trucks_capacity=350, speed_kph=48.3, operating_hours=12, num_trucks=550, min_capacity_threshold=50, depot_location=None):
        super(DeliveryRouteEnv, self).__init__()
        if phase == 'train':
            self.data = data
        elif phase == 'validation':
            self.data = validation_data
        elif phase == 'test':
            self.data = test_data
        self.phase = phase  
        self.data = self.preprocess_data()
        self.trucks_capacity = trucks_capacity
        self.speed_kph = speed_kph
        self.operating_hours = operating_hours
        self.num_trucks = num_trucks
        self.last_delivery_accuracy = 0
        self.num_features = 12  # Total number of features in the observation space
        self.early_delivery_bonus = 10  # Bonus for vompleting delivery before estimated date.
        self.penalty_per_km = -1  # Penalty for each kilometer traveled
        self.penalty_per_hour = -0.5  # Penalty for each hour spent
        self.penalty_per_excess_truck = -5  # Penalty for each excess truck used
        self.min_capacity_threshold = min_capacity_threshold  # Minimum capacity before considering return to depot
        self.action_space = spaces.Discrete(self.num_delivery_points)
        self.current_delivery_index = 0  # Track the current delivery being processed
        self.simulated_time = pd.to_datetime(self.data['order_purchase_timestamp_geo'].min())  # Initialize simulated time to the first order time
        self.total_operational_hours = 0  # Attribute for tracking operational hours
        self.operating_hours_exceeded = False
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_features,), dtype=np.float32)
        # Set the depot location
        if depot_location is not None:
            self.depot_location = depot_location
        else:
            self.depot_location = (self.data['seller_latitude_geo'].iloc[0], self.data['seller_longitude_geo'].iloc[0])

    def step(self, action):
        self.take_action(action)
        self.current_step += 1
        reward = self.calculate_reward()
        self.done = self.check_if_done()
        next_state = self.get_next_state()

        # Travel time calculation 
        travel_time_hours = self.last_distance_traveled / self.speed_kph

        service_time_hours = 0.5  

        # Actual delivery time calculation
        delivery_time_hours = travel_time_hours + service_time_hours

        # Retrieve the expected delivery time for the current order
        current_order = self.data.iloc[self.current_delivery_index]
        order_purchase_timestamp = pd.to_datetime(current_order['order_purchase_timestamp_geo'])
        expected_delivery_date = pd.to_datetime(current_order['order_estimated_delivery_date_geo'])

        # Calculate hours until the expected delivery date from the order purchase timestamp
        expected_delivery_time_hours = (expected_delivery_date - order_purchase_timestamp).total_seconds() / 3600.0

        # Determine on-time status based on whether the delivery_time_hours is less than or equal to expected_delivery_time_hours
        on_time = delivery_time_hours <= expected_delivery_time_hours

        # Update metrics based on the action's outcome
        self.total_distance_traveled += self.last_distance_traveled
        self.total_operational_hours += delivery_time_hours

        info = {
            'delivery_time_hours': delivery_time_hours,
            'on_time': on_time
        }

        return next_state, reward, self.done, info


    def reset(self):
        self.current_step = 0
        self.done = False
        self.total_distance_traveled = 0
        self.total_operational_hours = 0
        self.total_time_elapsed = 0
        self.current_truck_capacity = self.trucks_capacity
        initial_state = self.get_initial_state()
        self.current_location = (initial_state[0], initial_state[1])
        self.current_reward = 0
        self.last_distance_traveled = 0
        self.current_delivery_index = 0
        self.simulated_time = pd.to_datetime(self.data['order_purchase_timestamp_geo'].min())
        return initial_state
    
    def render(self, mode='console'):
        if mode == 'console':
            print(f"Current Location: {self.current_location}")
            print(f"Truck Capacity Remaining: {self.current_truck_capacity}")
            print(f"Total Distance Traveled: {self.total_distance_traveled} km")
            print(f"Total Time Elapsed: {self.total_time_elapsed} hours")
            print(f"Number of Deliveries Made: {self.current_step}")

    def preprocess_data(self):
        self.data.sort_values(by='order_estimated_delivery_date_geo', inplace=True)
        self.data['delivery_window'] = pd.to_datetime(self.data['order_estimated_delivery_date_geo']) + pd.Timedelta(hours=2)
        unique_points = self.data.groupby(['geolocation_lat_geo', 'geolocation_lng_geo']).ngroups
        self.num_delivery_points = unique_points
        self.data['order_hour'] = pd.to_datetime(self.data['order_purchase_timestamp_geo']).dt.hour
        # Initialize 'delivered' column
        self.data['delivered'] = False
        return self.data
    
    def calculate_reward(self):
        # Initialize the reward
        reward = 0
    
        # Add a bonus for each successful delivery
        reward += self.early_delivery_bonus
    
        # Apply penalties for increased distance traveled since last action
        distance_penalty = self.penalty_per_km * self.last_distance_traveled
        reward += distance_penalty
    
        # Apply penalties for late deliveries based on 'estimated_delivery_accuracy'
        if self.last_delivery_accuracy < 0:
            late_delivery_penalty = 10 * abs(self.last_delivery_accuracy)
            reward -= late_delivery_penalty
        return reward

    
    def identify_next_point(self, action):
        next_point_id = self.sorted_delivery_points[action]
        next_point_location = self.get_location(next_point_id, "delivery")
        return next_point_id, next_point_location

    def calculate_distance(self, from_location, to_location):
        lat1, lon1 = from_location
        lat2, lon2 = to_location
        return haversine(lon1, lat1, lon2, lat2)
    
    def get_location(self, point_id, point_type):
        # Retrieve the (latitude, longitude) of a point from the dataset
        if point_type == "delivery":
            location = self.data.loc[self.data['order_id'] == point_id, ['geolocation_lat_geo', 'geolocation_lng_geo']].values[0]
        else:  # Assuming "seller" for now
            location = self.data.loc[self.data['order_id'] == point_id, ['seller_latitude_geo', 'seller_longitude_geo']].values[0]
        return tuple(location)
    
    def check_delivery_time(self, order_id):
        # Retrieve the row corresponding to the order_id
        order_row = self.data.loc[self.data['order_id'] == order_id]
        # 'estimated_delivery_accuracy' is the difference between estimated delivery and actual delivery.
        # A positive value means the delivery was made earlier than estimated, and vice versa.
        estimated_delivery_accuracy = order_row['estimated_delivery_accuracy'].values[0]
        # If 'estimated_delivery_accuracy' is positive, the delivery was made before the estimated time.
        if estimated_delivery_accuracy > 0:
            return self.early_delivery_bonus
        return 0
    
    def update_simulated_time(self, from_location, to_location):
        # Existing logic to calculate travel time and update simulated_time
        distance_km = self.calculate_distance(from_location, to_location)
        travel_time_hours = distance_km / self.speed_kph
        self.simulated_time += pd.Timedelta(hours=travel_time_hours)
        # Check if the operating hours have been exceeded
        if self.simulated_time.hour >= self.operating_hours:
            self.operating_hours_exceeded = True
    
    def need_to_return_to_depot(self):
        # Decide based on truck's remaining capacity
        if self.current_truck_capacity <= self.min_capacity_threshold or self.time_since_last_depot_visit >= self.max_time_without_return:
            return True
        return False
        
    def return_to_depot(self):
        # Handle the process of returning to the depot
        self.current_truck_capacity = self.trucks_capacity  # Reset capacity
        # Update location to depot's location
        self.total_distance_traveled += self.calculate_distance(self.current_location, self.depot_location)
        self.current_location = self.depot_location

    def update_truck_status(self):
        self.current_truck_capacity -= 1  
        if self.need_to_return_to_depot():
            self.return_to_depot()

    def get_initial_state(self):
        initial_row = self.data.iloc[0]
        initial_state = np.array([
        initial_row['geolocation_lat_geo'],  
        initial_row['geolocation_lng_geo'],  
        initial_row['order_hour'],  
        self.current_truck_capacity,
        initial_row['distance_km'],
        initial_row['estimated_delivery_accuracy'],
        initial_row['delivery_duration'],
        initial_row['approval_duration'],
        initial_row['carrier_handling_duration'],
        initial_row['geocluster_id'],
        initial_row['tempcluster_id'],
        initial_row['customer_density']])
        return initial_state

    def take_action(self, action):
        if 'delivered' not in self.data.columns:
            self.data['delivered'] = False

        upcoming_deliveries = self.data[(self.data['delivery_window'] > self.simulated_time) & (~self.data['delivered'])].copy()

        if upcoming_deliveries.empty or self.current_truck_capacity <= 0 or self.simulated_time.hour >= self.operating_hours:
            if self.current_truck_capacity <= 0 or self.simulated_time.hour >= self.operating_hours:
                self.return_to_depot()
            self.done = True
            return self.get_next_state(), 0, self.done

        # Calculate scores directly within upcoming_deliveries DataFrame
        upcoming_deliveries['priority_score'] = 1 / (upcoming_deliveries['delivery_window'] - self.simulated_time).dt.total_seconds()
        upcoming_deliveries['proximity_score'] = 1 / upcoming_deliveries.apply(lambda row: self.calculate_distance(self.current_location, (row['geolocation_lat_geo'], row['geolocation_lng_geo'])), axis=1)
        upcoming_deliveries['combined_score'] = (0.7 * upcoming_deliveries['priority_score']) + (0.3 * upcoming_deliveries['proximity_score'])

        next_delivery_idx = upcoming_deliveries['combined_score'].idxmax()
        next_delivery = upcoming_deliveries.loc[next_delivery_idx]

        expected_delivery_date = pd.to_datetime(next_delivery['order_estimated_delivery_date_geo'])
        expected_delivery_time_hours = (expected_delivery_date - self.simulated_time).total_seconds() / 3600.0

        new_location = (next_delivery['geolocation_lat_geo'], next_delivery['geolocation_lng_geo'])
        self.last_distance_traveled = self.calculate_distance(self.current_location, new_location)

        service_time_hours = 0.5
        travel_time_hours = self.last_distance_traveled / self.speed_kph
        delivery_time_hours = travel_time_hours + service_time_hours

        self.last_delivery_accuracy = delivery_time_hours - expected_delivery_time_hours

        # Mark the delivery as completed in the original DataFrame
        self.data.loc[next_delivery_idx, 'delivered'] = True

        self.update_simulated_time(self.current_location, new_location)
        self.current_location = new_location

        if self.simulated_time.hour > self.operating_hours:
            self.operating_hours_exceeded = True

        self.current_truck_capacity -= 1
        if self.current_truck_capacity <= self.min_capacity_threshold:
            self.return_to_depot()

        reward = self.calculate_reward()
        next_state = self.get_next_state()

        return next_state, reward, self.done

    def get_next_state(self):
        # Check if the episode is not done and a next point has been identified
        if not self.done and hasattr(self, 'next_point_id') and self.next_point_id in self.data['order_id'].values:
            next_row = self.data.loc[self.data['order_id'] == self.next_point_id].iloc[0]
        else:
            next_row = self.data.iloc[0]  
    
        self.current_order_hour = self.simulated_time.hour
    
        # Constructing the next state with all desired features
        next_state = np.array([
            next_row['geolocation_lat_geo'],
            next_row['geolocation_lng_geo'],
            self.current_order_hour,  
            self.current_truck_capacity,
            next_row['distance_km'],  
            next_row['estimated_delivery_accuracy'],  
            next_row['delivery_duration'],  
            next_row['approval_duration'],  
            next_row['carrier_handling_duration'],  
            next_row['geocluster_id'],  
            next_row['tempcluster_id'],  
            next_row['customer_density'] ])
        return next_state


    def check_if_done(self):
        """
        Check if the episode is done, which could be when all deliveries for the day are completed,
        or the operating hours limit is reached.
        """
        if self.current_step >= len(self.data) - 1 or self.operating_hours_exceeded:
            return True
        return False

# Initialize environment
data_path = './data/rl_final_model_dataset.csv'
loaded_data = pd.read_csv(data_path)

# Load dataset from 'rl_final_model_dataset.csv'
train_data, test_data = train_test_split(loaded_data, test_size=0.3, random_state=42)
validation_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Initialize environments
train_env = DeliveryRouteEnv(data=train_data, phase='train')
validation_env = DeliveryRouteEnv(data=validation_data, phase='validation')
test_env = DeliveryRouteEnv(data=test_data, phase='test')

# Setup EvalCallback for periodic evaluation
eval_callback = EvalCallback(validation_env,
                             best_model_save_path='./logs/best_model',
                             log_path='./logs/eval',
                             eval_freq=500,
                             deterministic=True,
                             render=False)

# Initialize model with tensorboard logging
model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./ppo_delivery_route_tb/")

# Training the model with EvalCallback
model.learn(total_timesteps=40000, callback=[eval_callback])

# Save the model
model.save("ppo_delivery_route_optimization")

def evaluate_model(env, model, num_episodes=100):
    total_rewards = 0
    total_distance_traveled = 0
    total_operational_hours = 0
    total_deliveries = 0
    on_time_deliveries = 0
    total_delivery_time = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            total_rewards += rewards
            # Accessing metrics directly from the environment or info dictionary
            total_distance_traveled += env.total_distance_traveled
            total_operational_hours += env.total_operational_hours
            if 'delivery_time' in info and 'on_time' in info:
                total_delivery_time += info['delivery_time']
                total_deliveries += 1
                if info['on_time']:
                    on_time_deliveries += 1

    avg_reward = total_rewards / num_episodes
    avg_distance_traveled = total_distance_traveled / num_episodes
    avg_operational_hours = total_operational_hours / num_episodes
    avg_delivery_time = total_delivery_time / total_deliveries if total_deliveries > 0 else 0
    on_time_delivery_rate = (on_time_deliveries / total_deliveries) * 100 if total_deliveries > 0 else 0
    
    print(f"Average Reward: {avg_reward}")
    print(f"Average Distance Traveled: {avg_distance_traveled} km")
    print(f"Average Operational Hours: {avg_operational_hours}")
    print(f"Average Delivery Time: {avg_delivery_time}")
    print(f"On-Time Delivery Rate: {on_time_delivery_rate}%")

    return avg_reward, avg_distance_traveled, avg_operational_hours, avg_delivery_time, on_time_delivery_rate

# Evaluate the model using evaluate_model.
evaluate_model(train_env, model, num_episodes=100)
evaluate_model(validation_env, model, num_episodes=100)
evaluate_model(test_env, model, num_episodes=100)
