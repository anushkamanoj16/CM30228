Packages installed.
Starting data preparation...
Missing values in each column after all deletions:
 order_id                          0
customer_id                       0
order_status                      0
order_purchase_timestamp          0
order_approved_at                13
order_delivered_carrier_date      1
order_delivered_customer_date     0
order_estimated_delivery_date     0
customer_unique_id                0
customer_zip_code_prefix          0
customer_city                     0
customer_state                    0
geolocation_lat                   0
geolocation_lng                   0
seller_id                         0
seller_latitude                   0
seller_longitude                  0
dtype: int64

Shape of the final dataset without missing values or duplicates:  (88862, 17)
                           order_id                       customer_id  ... seller_latitude seller_longitude
0  e481f51cbdc54678b7cc49136f2d6af7  9ef432eb6251297304e76186b10a928d  ...      -23.666558       -46.459914
1  53cdb2fc8bc7dce0b6741e2150273451  b0830fb4747a6c6d20dea0b8c802d7ef  ...      -22.492669       -47.829376
2  47770eb9100c2d0c44946d9cf07ec65d  41ce2a54c0b03bf3443c3d931a367089  ...      -21.360188       -48.228224
3  949d5b44dbf5de918fe9c16f97b45f8a  f88197465ea7920adcdbec7375364d82  ...      -19.919052       -43.938668
4  ad21c59c0840e6cb83a9ceb5573f8159  8ab97904e6daea8866dbdbc4fb7aad2c  ...      -23.521727       -46.186005

[5 rows x 17 columns]
Data preparation complete.
Preparing temporal and geospatial features...
Features prepared.
Performing clustering...
./data/splits/tempclustering_train_dataset.csv - Silhouette Score: 0.559885837682999
./data/splits/tempclustering_validation_dataset.csv - Silhouette Score: 0.5367919037542586
./data/splits/tempclustering_test_dataset.csv - Silhouette Score: 0.5339232783278238
./data/splits/geoclustering_train_dataset.csv - Silhouette Score: 0.5549780962649309
./data/splits/geoclustering_validation_dataset.csv - Silhouette Score: 0.5505256344498339
./data/splits/geoclustering_test_dataset.csv - Silhouette Score: 0.5500778592929028
Clustering complete.
Preparing features for reinforcement learning...
NaN values report for each column:
order_id                             0
geolocation_lat_geo                  0
geolocation_lng_geo                  0
seller_latitude_geo                  0
seller_longitude_geo                 0
distance_km                          0
geocluster_id                        0
customer_state_geo                   0
order_purchase_timestamp_geo         0
estimated_delivery_accuracy          0
delivery_duration                    0
order_estimated_delivery_date_geo    0
approval_duration                    0
carrier_handling_duration            0
tempcluster_id                       0
customer_density                     0
order_delivered_customer_date_geo    0
dtype: int64

No NaN values found.
RL features ready.
Running the reinforcement learning model...
/Users/anushkamanoj/Library/Python/3.8/lib/python/site-packages/stable_baselines3/common/vec_env/patch_gym.py:49: UserWarning: You provided an OpenAI Gym environment. We strongly recommend transitioning to Gymnasium environments. Stable-Baselines3 is automatically wrapping your environments in a compatibility layer, which could potentially cause issues.
  warnings.warn(
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
Logging to ./ppo_delivery_route_tb/PPO_1
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
Logging to ./ppo_delivery_route_tb/PPO_1
Eval num_timesteps=500, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 500      |
---------------------------------
New best mean reward!
Eval num_timesteps=1000, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 1000     |
---------------------------------
Eval num_timesteps=1500, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 1500     |
---------------------------------
Eval num_timesteps=2000, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 2000     |
---------------------------------
----------------------------------
| rollout/           |           |
|    ep_len_mean     | 1         |
|    ep_rew_mean     | -1.11e+05 |
| time/              |           |
|    fps             | 0         |
|    iterations      | 1         |
|    time_elapsed    | 7727      |
|    total_timesteps | 2048      |
----------------------------------
Eval num_timesteps=2500, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------------
| eval/                   |           |
|    mean_ep_length       | 1.59e+04  |
|    mean_reward          | 1.59e+05  |
| time/                   |           |
|    total_timesteps      | 2500      |
| train/                  |           |
|    approx_kl            | 0.0       |
|    clip_fraction        | 0         |
|    clip_range           | 0.2       |
|    entropy_loss         | -9.5      |
|    explained_variance   | 1.19e-07  |
|    learning_rate        | 0.0003    |
|    loss                 | 4.2e+11   |
|    n_updates            | 10        |
|    policy_gradient_loss | -8.51e-06 |
|    value_loss           | 8.29e+11  |
---------------------------------------
Eval num_timesteps=3000, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 3000     |
---------------------------------
Eval num_timesteps=3500, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 3500     |
---------------------------------
Eval num_timesteps=4000, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 4000     |
---------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1        |
|    ep_rew_mean     | -1.2e+05 |
| time/              |          |
|    fps             | 0        |
|    iterations      | 2        |
|    time_elapsed    | 14666    |
|    total_timesteps | 4096     |
---------------------------------
Eval num_timesteps=4500, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------------
| eval/                   |           |
|    mean_ep_length       | 1.59e+04  |
|    mean_reward          | 1.59e+05  |
| time/                   |           |
|    total_timesteps      | 4500      |
| train/                  |           |
|    approx_kl            | 0.0       |
|    clip_fraction        | 0         |
|    clip_range           | 0.2       |
|    entropy_loss         | -9.5      |
|    explained_variance   | 0         |
|    learning_rate        | 0.0003    |
|    loss                 | 8.31e+09  |
|    n_updates            | 20        |
|    policy_gradient_loss | -3.54e-05 |
|    value_loss           | 1.54e+10  |
---------------------------------------
Eval num_timesteps=5000, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 5000     |
---------------------------------
Eval num_timesteps=5500, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 5500     |
---------------------------------
Eval num_timesteps=6000, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 6000     |
---------------------------------
----------------------------------
| rollout/           |           |
|    ep_len_mean     | 1         |
|    ep_rew_mean     | -1.13e+05 |
| time/              |           |
|    fps             | 0         |
|    iterations      | 3         |
|    time_elapsed    | 148103    |
|    total_timesteps | 6144      |
----------------------------------
Eval num_timesteps=6500, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
--------------------------------------
| eval/                   |          |
|    mean_ep_length       | 1.59e+04 |
|    mean_reward          | 1.59e+05 |
| time/                   |          |
|    total_timesteps      | 6500     |
| train/                  |          |
|    approx_kl            | 0.0      |
|    clip_fraction        | 0        |
|    clip_range           | 0.2      |
|    entropy_loss         | -9.5     |
|    explained_variance   | 1.19e-07 |
|    learning_rate        | 0.0003   |
|    loss                 | 7.97e+09 |
|    n_updates            | 30       |
|    policy_gradient_loss | -3.5e-05 |
|    value_loss           | 1.56e+10 |
--------------------------------------
Eval num_timesteps=7000, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 7000     |
---------------------------------
Eval num_timesteps=7500, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 7500     |
---------------------------------
Eval num_timesteps=8000, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 8000     |
---------------------------------
----------------------------------
| rollout/           |           |
|    ep_len_mean     | 1         |
|    ep_rew_mean     | -1.22e+05 |
| time/              |           |
|    fps             | 0         |
|    iterations      | 4         |
|    time_elapsed    | 154090    |
|    total_timesteps | 8192      |
----------------------------------
Eval num_timesteps=8500, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------------
| eval/                   |           |
|    mean_ep_length       | 1.59e+04  |
|    mean_reward          | 1.59e+05  |
| time/                   |           |
|    total_timesteps      | 8500      |
| train/                  |           |
|    approx_kl            | 0.0       |
|    clip_fraction        | 0         |
|    clip_range           | 0.2       |
|    entropy_loss         | -9.5      |
|    explained_variance   | 0         |
|    learning_rate        | 0.0003    |
|    loss                 | 8.51e+09  |
|    n_updates            | 40        |
|    policy_gradient_loss | -3.49e-05 |
|    value_loss           | 1.57e+10  |
---------------------------------------
Eval num_timesteps=9000, episode_reward=158980.00 +/- 0.00
Episode length: 15898.00 +/- 0.00
---------------------------------
| eval/              |          |
|    mean_ep_length  | 1.59e+04 |
|    mean_reward     | 1.59e+05 |
| time/              |          |
|    total_timesteps | 9000     |
---------------------------------

Please note that this is not the full run as due to computational and temporla constraints this was not possible to be fully run.
