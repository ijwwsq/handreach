import gymnasium as gym
import panda_gym
import numpy as np
import cv2
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from collections import deque
import os
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== КОНФИГУРАЦИЯ ====================
ENV_NAME = "PandaReach-v3"
MAX_EPISODE_STEPS = 50
TARGET_RANGE = 0.3
WINDOW_SCALE = 1.5
MODEL_DIR = "./models/"
LOG_DIR = "./logs/"
PLOT_DIR = "./plots/"

os.makedirs(PLOT_DIR, exist_ok=True)

# ==================== ЦВЕТА ====================
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_DARK = (30, 30, 30)

# ==================== ИНИЦИАЛИЗАЦИЯ ====================
print("=" * 70)
print("PANDA REACH - VISUALIZATION & EVALUATION")
print("=" * 70)
print(f"\nLoading model from {MODEL_DIR}...")
model = PPO.load(MODEL_DIR + "best_model")
print("Model loaded successfully")

def make_env():
    env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
    env.unwrapped.target_range = TARGET_RANGE
    return env

test_env = DummyVecEnv([make_env])

# Загрузка нормализации
norm_files = ["vec_normalize_final.pkl", "vec_normalize_interrupted.pkl", "vec_normalize.pkl"]
norm_loaded = False
for norm_file in norm_files:
    norm_path = MODEL_DIR + norm_file
    if os.path.exists(norm_path):
        print(f"Loading normalization: {norm_file}")
        test_env = VecNormalize.load(norm_path, test_env)
        test_env.training = False
        test_env.norm_reward = False
        norm_loaded = True
        print("Normalization loaded successfully")
        break

if not norm_loaded:
    print("WARNING: Normalization not found - continuing without it")

print("\nStarting visualization...")
print("Controls: [Q] - quit | [R] - reset episode | [S] - save plots")
print("=" * 70 + "\n")

# ==================== СТАТИСТИКА ====================
class Statistics:
    def __init__(self):
        self.episode_count = 0
        self.success_count = 0
        self.total_reward = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.reward_history = deque(maxlen=10)
        self.success_history = deque(maxlen=10)
        
    def update_episode(self, reward, steps, success):
        self.episode_count += 1
        self.total_reward += reward
        self.total_steps += steps
        self.episode_rewards.append(reward)
        self.episode_lengths.append(steps)
        self.episode_successes.append(1 if success else 0)
        self.reward_history.append(reward)
        if success:
            self.success_count += 1
            self.success_history.append(1)
        else:
            self.success_history.append(0)
    
    def get_avg_reward(self):
        return np.mean(self.reward_history) if self.reward_history else 0
    
    def get_success_rate(self):
        if self.episode_count == 0:
            return 0
        return (self.success_count / self.episode_count) * 100
    
    def get_recent_success_rate(self):
        if not self.success_history:
            return 0
        return (sum(self.success_history) / len(self.success_history)) * 100
    
    def get_avg_episode_length(self):
        if not self.episode_lengths:
            return 0
        return np.mean(self.episode_lengths)
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Total Episodes:          {self.episode_count}")
        print(f"Successful Episodes:     {self.success_count}")
        print(f"Success Rate:            {self.get_success_rate():.2f}%")
        print(f"Average Reward:          {self.total_reward / max(self.episode_count, 1):.3f}")
        print(f"Average Episode Length:  {self.get_avg_episode_length():.1f} steps")
        print(f"Total Steps:             {self.total_steps}")
        if self.episode_rewards:
            print(f"Best Reward:             {max(self.episode_rewards):.3f}")
            print(f"Worst Reward:            {min(self.episode_rewards):.3f}")
            print(f"Reward Std Dev:          {np.std(self.episode_rewards):.3f}")
        print("=" * 70 + "\n")
    
    def save_plots(self, save_dir):
        if self.episode_count == 0:
            print("No data to plot")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Panda Reach Evaluation Results', fontsize=16, fontweight='bold')
        
        # Episode Rewards
        ax1 = axes[0, 0]
        episodes = range(1, len(self.episode_rewards) + 1)
        ax1.plot(episodes, self.episode_rewards, 'b-', alpha=0.6, linewidth=1)
        if len(self.episode_rewards) > 5:
            window = min(10, len(self.episode_rewards) // 3)
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(self.episode_rewards) + 1), moving_avg, 'r-', linewidth=2, label=f'MA({window})')
            ax1.legend()
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Episode Rewards')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Success Rate Over Time
        ax2 = axes[0, 1]
        cumulative_success_rate = [np.mean(self.episode_successes[:i+1]) * 100 for i in range(len(self.episode_successes))]
        ax2.plot(episodes, cumulative_success_rate, 'g-', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Cumulative Success Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50%')
        ax2.legend()
        
        # Episode Lengths
        ax3 = axes[1, 0]
        ax3.plot(episodes, self.episode_lengths, 'purple', alpha=0.6, linewidth=1)
        if len(self.episode_lengths) > 5:
            window = min(10, len(self.episode_lengths) // 3)
            moving_avg = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            ax3.plot(range(window, len(self.episode_lengths) + 1), moving_avg, 'orange', linewidth=2, label=f'MA({window})')
            ax3.legend()
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.set_title('Episode Length')
        ax3.grid(True, alpha=0.3)
        
        # Reward Distribution
        ax4 = axes[1, 1]
        ax4.hist(self.episode_rewards, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(x=np.mean(self.episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Reward Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = os.path.join(save_dir, f'evaluation_results_{timestamp}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {filename}")
        plt.close()

stats = Statistics()

# ==================== ВИЗУАЛИЗАЦИЯ ====================
def create_overlay(frame, episode_reward, step, stats, is_success, fps):
    """Создание информационного оверлея"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Полупрозрачная панель сверху
    cv2.rectangle(overlay, (0, 0), (w, 180), COLOR_DARK, -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Вся информация белым цветом
    # Заголовок
    cv2.putText(frame, "PANDA REACH AGENT", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
    
    # Левая колонка
    cv2.putText(frame, f"Episode: {stats.episode_count + 1}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    cv2.putText(frame, f"Step: {step}/{MAX_EPISODE_STEPS}", (10, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    cv2.putText(frame, f"Reward: {episode_reward:.2f}", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    cv2.putText(frame, f"Avg Reward: {stats.get_avg_reward():.2f}", (10, 135), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    # Правая колонка
    cv2.putText(frame, f"Success: {stats.success_count}/{stats.episode_count}", (w - 250, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    cv2.putText(frame, f"Success Rate: {stats.get_success_rate():.1f}%", (w - 250, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)
    
    cv2.putText(frame, f"Recent (10): {stats.get_recent_success_rate():.1f}%", (w - 250, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    # Большое уведомление об успехе
    if is_success:
        cv2.rectangle(overlay, (w//2 - 150, h//2 - 50), (w//2 + 150, h//2 + 50), 
                     COLOR_GREEN, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        cv2.putText(frame, "SUCCESS!", (w//2 - 120, h//2 + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, COLOR_WHITE, 4)
    
    return frame

# ==================== ГЛАВНЫЙ ЦИКЛ ====================
obs = test_env.reset()
episode_reward = 0
step = 0
prev_time = time.time()

try:
    while True:
        # Получение действия и выполнение шага
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        episode_reward += reward[0]
        step += 1
        
        # Рендеринг кадра
        frame = test_env.envs[0].render()
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.resize(
            frame, 
            (int(frame.shape[1] * WINDOW_SCALE), 
             int(frame.shape[0] * WINDOW_SCALE)),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Расчет FPS
        fps = 1 / max(time.time() - prev_time, 1e-6)
        prev_time = time.time()
        
        # Добавление информационного оверлея
        is_success = info[0].get('is_success', False)
        display_frame = create_overlay(frame, episode_reward, step, stats, is_success, fps)
        
        # Отображение
        cv2.imshow("Panda Reach", display_frame)
        
        # Завершение эпизода
        if done[0]:
            stats.update_episode(episode_reward, step, is_success)
            
            status = "SUCCESS" if is_success else "FAILED "
            print(f"Episode {stats.episode_count:4d} | {status} | "
                  f"Reward: {episode_reward:7.3f} | "
                  f"Steps: {step:2d} | "
                  f"Success Rate: {stats.get_success_rate():6.2f}%")
            
            obs = test_env.reset()
            episode_reward = 0
            step = 0
        
        # Обработка клавиш
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\nResetting episode...")
            obs = test_env.reset()
            episode_reward = 0
            step = 0
        elif key == ord('s'):
            print("\nSaving plots...")
            stats.save_plots(PLOT_DIR)

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

finally:
    # Финальная статистика и графики
    stats.print_summary()
    
    if stats.episode_count > 0:
        print("Generating final plots...")
        stats.save_plots(PLOT_DIR)
    
    test_env.close()
    cv2.destroyAllWindows()