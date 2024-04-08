
# silhouette_scores = []
# for k in range(2, 11):  # Silhouette score cannot be calculated for a single cluster
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(train_data)
#     score = silhouette_score(train_data, kmeans.labels_)
#     silhouette_scores.append(score)

# plt.plot(range(2, 11), silhouette_scores, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Method')
# plt.show()

# sse = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42).fit(train_data)
#     sse.append(kmeans.inertia_)

# plt.plot(range(1, 11), sse, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('SSE')
# plt.title('Elbow Method')
# plt.show()

# def take_action(self, action):
#         """
#         Execute the chosen action, which involves moving to the next delivery point and updating the environment state.
#         This method now includes updating the elapsed time based on travel distance and speed, managing truck capacity,
#         and basic delivery logic including handling of delivery time.
#         """
#         # Identify the next delivery point and its location based on the action taken
#         next_point_id, next_point_location = self._identify_next_point(action)

#         # Calculate the distance to the next point and update the total distance traveled
#         distance_traveled = self._calculate_distance(self.current_location, next_point_location)
#         self.total_distance_traveled += distance_traveled

#         # Calculate the time taken to travel to the next point based on distance and truck speed
#         # Speed is in km/h, so distance (km) / speed (km/h) gives time in hours
#         time_taken = distance_traveled / self.speed_kph
#         self.total_time_elapsed += time_taken

#         # Update the current location to the next point's location
#         self.current_location = next_point_location

#         # Decrease the truck's capacity by 1 for each delivery made
#         # Assuming each delivery action corresponds to a single package delivery
#         self.current_truck_capacity -= 1

#         # If the truck's capacity reaches 0 or a critical low threshold, it may need to return to a depot
#         if self.current_truck_capacity <= 0 or self._need_to_return_to_depot():
#             self._return_to_depot()

#         # Update the delivery status in your dataset, marking the order as delivered
#         # This step assumes your data structure can track delivery status
#         self.data.loc[self.data['order_id'] == next_point_id, 'delivery_status'] = 'Delivered'

#         # Delivery logic: Check if the delivery was made on time, early, or late
#         # and apply any logic related to delivery timing, such as bonuses for early delivery
#         reward_bonus = self._check_delivery_time(next_point_id)
#         self.current_reward += reward_bonus

#         # Check if operating hours limit is reached, ending the day's delivery operations if so
#         if self.total_time_elapsed >= self.operating_hours:
#             self.done = True