import gymnasium as gym
import panda_gym
import numpy as np
import cv2
import time
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==================== CONFIGURATION ====================
ENV_NAME = "PandaReach-v3"
MAX_EPISODE_STEPS = 50
TARGET_RANGE = 0.3
TOTAL_TIMESTEPS = 300_000
WINDOW_SCALE = 1.5
N_ENVS = 4
EVAL_FREQ = 5000
N_EVAL_EPISODES = 10
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
PLOT_DIR = "./plots/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ==================== ENVIRONMENT CREATION ====================
def make_env(rank, seed=0):
    """
    Creates a single environment instance with unique seeding.
    
    Args:
        rank: Environment rank for parallel execution
        seed: Base random seed
    
    Returns:
        Initialized environment wrapped in Monitor
    """
    def _init():
        env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
        env.unwrapped.target_range = TARGET_RANGE
        env = Monitor(env, LOG_DIR + f"env_{rank}")
        env.reset(seed=seed + rank)
        return env
    return _init

print("=" * 70)
print("PANDA REACH - TRAINING PIPELINE")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Environment:        {ENV_NAME}")
print(f"  Parallel Envs:      {N_ENVS}")
print(f"  Total Timesteps:    {TOTAL_TIMESTEPS:,}")
print(f"  Max Episode Steps:  {MAX_EPISODE_STEPS}")
print(f"  Evaluation Freq:    {EVAL_FREQ:,}")
print(f"  Target Range:       {TARGET_RANGE}")
print("=" * 70)

# Create vectorized training environment
print("\nInitializing training environment...")
train_env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])

# Apply normalization for training stability
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99
)
print("Training environment initialized with normalization")

# Create evaluation environment
print("Initializing evaluation environment...")
eval_env = DummyVecEnv([make_env(0, seed=100)])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
    training=False
)
print("Evaluation environment initialized")

# ==================== CALLBACKS ====================
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=0
)

checkpoint_callback = CheckpointCallback(
    save_freq=EVAL_FREQ,
    save_path=MODEL_DIR,
    name_prefix="panda_reach_checkpoint",
    verbose=0
)

# ==================== MODEL INITIALIZATION ====================
print("\nInitializing PPO model...")
print("Hyperparameters:")
print("  Learning Rate:      3e-4")
print("  Steps per Update:   2048")
print("  Batch Size:         64")
print("  Epochs:             10")
print("  Gamma:              0.99")
print("  GAE Lambda:         0.95")

model = PPO(
    "MultiInputPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log=LOG_DIR
)

# Alternative: SAC (often superior for robotics tasks)
# model = SAC(
#     "MultiInputPolicy",
#     train_env,
#     learning_rate=3e-4,
#     buffer_size=1_000_000,
#     learning_starts=1000,
#     batch_size=256,
#     tau=0.005,
#     gamma=0.99,
#     train_freq=1,
#     gradient_steps=1,
#     verbose=1,
#     tensorboard_log=LOG_DIR
# )

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
start_time = time.time()

# ==================== TRAINING LOOP ====================
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False
    )
    
    training_time = time.time() - start_time
    
    # Save final model and normalization
    model.save(MODEL_DIR + "panda_reach_final")
    train_env.save(MODEL_DIR + "vec_normalize_final.pkl")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Total Training Time:  {training_time:.1f}s ({training_time/60:.1f}m)")
    print(f"Best Model Path:      {MODEL_DIR}best_model.zip")
    print(f"Final Model Path:     {MODEL_DIR}panda_reach_final.zip")
    print("=" * 70)
    
except KeyboardInterrupt:
    training_time = time.time() - start_time
    print("\n\nTraining interrupted by user")
    print(f"Training Time: {training_time:.1f}s")
    model.save(MODEL_DIR + "panda_reach_interrupted")
    train_env.save(MODEL_DIR + "vec_normalize_interrupted.pkl")
    print("Interrupted model saved")

# ==================== TRAINING RESULTS VISUALIZATION ====================
print("\nGenerating training plots...")
try:
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    
    results = load_results(LOG_DIR)
    x, y = ts2xy(results, 'timesteps')
    
    if len(y) > 0:
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Panda Reach Training Results', fontsize=16, fontweight='bold')
        
        # Episode Rewards
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(x, y, alpha=0.6, linewidth=1, color='blue')
        if len(y) > 50:
            window = 50
            moving_avg = np.convolve(y, np.ones(window)/window, mode='valid')
            ax1.plot(x[window-1:], moving_avg, 'r-', linewidth=2, label=f'MA({window})')
            ax1.legend()
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Episode Reward")
        ax1.set_title("Training Reward Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Episode Length
        ax2 = plt.subplot(2, 2, 2)
        episode_lengths = results['l'].values
        ax2.plot(episode_lengths, alpha=0.6, linewidth=1, color='purple')
        if len(episode_lengths) > 50:
            window = 50
            moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(episode_lengths)), moving_avg, 'orange', linewidth=2, label=f'MA({window})')
            ax2.legend()
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episode Length")
        ax2.set_title("Episode Length Over Time")
        ax2.grid(True, alpha=0.3)
        
        # Reward Distribution
        ax3 = plt.subplot(2, 2, 3)
        ax3.hist(y, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(x=np.mean(y), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(y):.2f}')
        ax3.axvline(x=np.median(y), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(y):.2f}')
        ax3.set_xlabel("Reward")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Reward Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Cumulative Reward
        ax4 = plt.subplot(2, 2, 4)
        cumulative_reward = np.cumsum(y)
        ax4.plot(cumulative_reward, color='green', linewidth=2)
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Cumulative Reward")
        ax4.set_title("Cumulative Reward Over Time")
        ax4.grid(True, alpha=0.3)
        ax4.ticklabel_format(style='plain', axis='y')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"{PLOT_DIR}training_results_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.savefig(LOG_DIR + "training_results.png", dpi=150, bbox_inches='tight')
        print(f"Training plots saved to: {plot_path}")
        plt.close()
        
        # Print training statistics
        print("\n" + "=" * 70)
        print("TRAINING STATISTICS")
        print("=" * 70)
        print(f"Total Episodes:       {len(y)}")
        print(f"Mean Reward:          {np.mean(y):.3f}")
        print(f"Std Reward:           {np.std(y):.3f}")
        print(f"Min Reward:           {np.min(y):.3f}")
        print(f"Max Reward:           {np.max(y):.3f}")
        print(f"Median Reward:        {np.median(y):.3f}")
        print(f"Mean Episode Length:  {np.mean(episode_lengths):.1f}")
        print("=" * 70)
    
except Exception as e:
    print(f"Failed to generate plots: {e}")

# ==================== MODEL EVALUATION ====================
print("\n" + "=" * 70)
print("STARTING EVALUATION")
print("=" * 70)

# Load best model
try:
    model = PPO.load(MODEL_DIR + "best_model")
    # For SAC: model = SAC.load(MODEL_DIR + "best_model")
    print("Best model loaded successfully")
except:
    print("Using current model for evaluation")

# Create test environment
test_env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
test_env.unwrapped.target_range = TARGET_RANGE

# Load normalization
try:
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecNormalize.load(MODEL_DIR + "vec_normalize_final.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False
    print("Normalization loaded successfully")
except:
    test_env = DummyVecEnv([lambda: test_env])
    print("WARNING: Normalization not loaded")

print("\nControls: [Q] - quit | [R] - reset episode")
print("=" * 70 + "\n")

obs = test_env.reset()
episode_reward = 0
episode_count = 0
success_count = 0
step = 0
prev_time = time.time()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    episode_reward += reward[0]
    step += 1
    
    # Render frame
    frame = test_env.envs[0].render()
    frame = (frame * 255).astype(np.uint8)
    frame = cv2.resize(
        frame,
        (int(frame.shape[1] * WINDOW_SCALE), int(frame.shape[0] * WINDOW_SCALE)),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Calculate FPS
    fps = 1 / max(time.time() - prev_time, 1e-6)
    prev_time = time.time()
    
    is_success = info[0].get('is_success', False)
    
    # Overlay information (all white text)
    cv2.putText(frame, f"Episode: {episode_count + 1}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Step: {step}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Reward: {episode_reward:.2f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if episode_count > 0:
        success_rate = (success_count / episode_count) * 100
        cv2.putText(frame, f"Success Rate: {success_rate:.1f}%", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Success Rate: -", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if is_success:
        cv2.putText(frame, "SUCCESS!", (frame.shape[1]//2 - 100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    cv2.imshow("Panda Reach - Evaluation", frame)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        obs = test_env.reset()
        episode_reward = 0
        step = 0
        continue
    
    if done[0]:
        if is_success:
            success_count += 1
        episode_count += 1
        
        status = "SUCCESS" if is_success else "FAILED "
        success_rate = (success_count / episode_count) * 100
        print(f"Episode {episode_count:4d} | {status} | "
              f"Reward: {episode_reward:7.3f} | "
              f"Steps: {step:2d} | "
              f"Success Rate: {success_rate:6.2f}%")
        
        obs = test_env.reset()
        episode_reward = 0
        step = 0

test_env.close()
cv2.destroyAllWindows()

# ==================== FINAL STATISTICS ====================
print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)
print(f"Total Episodes:       {episode_count}")
print(f"Successful Episodes:  {success_count}")
if episode_count > 0:
    print(f"Success Rate:         {(success_count/episode_count)*100:.2f}%")
else:
    print(f"Success Rate:         N/A")
print("=" * 70)