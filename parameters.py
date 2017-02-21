# for image
dims = (100, 100)
obs_height = 100
obs_width = 100
obs_channel = 1
history_number = 1

# for trainning
jobs = 1
max_iter_number = 7000
paths_number = 1
max_path_length = 200
batch_size = max_path_length
max_kl = 0.01
gae_lambda = 1.0
subsample_factor = 0.8
cg_damping = 0.001
discount = 0.99
cg_iters = 20
deviation = 0.1
render = False
train_flag = True
iter_num_per_train = 1
# checkpoint_file = "/home/air/trpo-master/checkpoint/FaceSeacher-v3-2000.ckpt"
checkpoint_file = None
iteration_number = 0


save_model_times = 500
record_movie = False
upload_to_gym = False
checkpoint_dir="checkpoint/"

# for environment

# environment_name = "MountainCarContinuous-v0"
# environment_name = "Pendulum-v0"
environment_name = "FaceSeacher-v3"
# for continous action
min_std = 3
center_adv = True
positive_adv = False
use_std_network = False
std = 1.1
obs_shape = 3
action_shape = 2
min_a = -1.0
max_a = 1.0

