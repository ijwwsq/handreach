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

# -----------------------------
# CONFIG
# -----------------------------
ENV_NAME = "PandaReach-v3"
MAX_EPISODE_STEPS = 50  # Короче для reach задачи - быстрее учится
TARGET_RANGE = 0.3
TOTAL_TIMESTEPS = 300_000  # Достаточно для reach
WINDOW_SCALE = 1.5
N_ENVS = 4  # Параллельные окружения для быстрого обучения
EVAL_FREQ = 5000
N_EVAL_EPISODES = 10
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# CREATE VECTORIZED TRAIN ENV
# -----------------------------
def make_env(rank, seed=0):
    """Создает одно окружение с уникальным seed"""
    def _init():
        env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
        env.unwrapped.target_range = TARGET_RANGE
        env = Monitor(env, LOG_DIR + f"env_{rank}")
        env.reset(seed=seed + rank)
        return env
    return _init

# Создаем векторизованное окружение
train_env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])

# Нормализация наблюдений и наград - критично для стабильности!
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99
)

# -----------------------------
# CREATE EVAL ENV
# -----------------------------
eval_env = DummyVecEnv([make_env(0, seed=100)])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,  # Не нормализуем награды при оценке
    clip_obs=10.0,
    training=False  # Режим оценки
)

# -----------------------------
# CALLBACKS
# -----------------------------
# Сохранение лучшей модели
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False
)

# Периодическое сохранение чекпоинтов
checkpoint_callback = CheckpointCallback(
    save_freq=EVAL_FREQ,
    save_path=MODEL_DIR,
    name_prefix="panda_reach_checkpoint"
)

# -----------------------------
# CREATE MODEL - ОПТИМАЛЬНЫЕ ГИПЕРПАРАМЕТРЫ
# -----------------------------
model = PPO(
    "MultiInputPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,  # Больше шагов перед обновлением
    batch_size=64,  # Оптимальный размер батча
    n_epochs=10,  # Эпох на обновление
    gamma=0.99,  # Discount factor
    gae_lambda=0.95,  # GAE lambda
    clip_range=0.2,  # PPO clip range
    ent_coef=0.0,  # Энтропийный бонус (можно увеличить до 0.01 для exploration)
    vf_coef=0.5,  # Value function coefficient
    max_grad_norm=0.5,  # Gradient clipping
    verbose=1,
    tensorboard_log=LOG_DIR
)

# Альтернатива: SAC часто лучше для робототехники
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

print("=" * 50)
print("НАЧИНАЕМ ОБУЧЕНИЕ")
print(f"Окружений: {N_ENVS}")
print(f"Всего timesteps: {TOTAL_TIMESTEPS}")
print(f"Оценка каждые: {EVAL_FREQ} шагов")
print("=" * 50)

# -----------------------------
# LEARN
# -----------------------------
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False  # Отключаем progress bar если нет tqdm
    )
    
    # Сохраняем финальную модель
    model.save(MODEL_DIR + "panda_reach_final")
    train_env.save(MODEL_DIR + "vec_normalize_final.pkl")
    
    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Лучшая модель сохранена в: {MODEL_DIR}best_model.zip")
    print("=" * 50)
    
except KeyboardInterrupt:
    print("\n\nОбучение прервано пользователем")
    model.save(MODEL_DIR + "panda_reach_interrupted")
    train_env.save(MODEL_DIR + "vec_normalize_interrupted.pkl")

# -----------------------------
# PLOT REWARDS
# -----------------------------
try:
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    
    results = load_results(LOG_DIR)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Episode rewards
    x, y = ts2xy(results, 'timesteps')
    ax1.plot(x, y)
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Training Reward")
    ax1.grid(True, alpha=0.3)
    
    # Moving average
    window_size = 50
    if len(y) > window_size:
        moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(moving_avg)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Average Reward")
        ax2.set_title(f"Moving Average (window={window_size})")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(LOG_DIR + "training_results.png", dpi=150)
    plt.show()
    print(f"\nГрафики сохранены в: {LOG_DIR}training_results.png")
    
except Exception as e:
    print(f"Не удалось построить графики: {e}")

# -----------------------------
# TESTING / VISUALIZATION
# -----------------------------
print("\n" + "=" * 50)
print("ЗАПУСК ВИЗУАЛИЗАЦИИ")
print("Нажмите 'q' для выхода, 'r' для сброса")
print("=" * 50 + "\n")

# Загружаем лучшую модель
try:
    model = PPO.load(MODEL_DIR + "best_model")
    # Для SAC: model = SAC.load(MODEL_DIR + "best_model")
    print("✓ Загружена лучшая модель")
except:
    print("⚠ Используем текущую модель")

# Создаем тестовое окружение
test_env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
test_env.unwrapped.target_range = TARGET_RANGE

# Загружаем нормализацию
try:
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecNormalize.load(MODEL_DIR + "vec_normalize_final.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False
    print("✓ Загружена нормализация\n")
except:
    test_env = DummyVecEnv([lambda: test_env])
    print("⚠ Нормализация не загружена\n")

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
    
    # Получаем кадр
    frame = test_env.envs[0].render()
    frame = (frame * 255).astype(np.uint8)
    frame = cv2.resize(
        frame,
        (int(frame.shape[1] * WINDOW_SCALE), int(frame.shape[0] * WINDOW_SCALE)),
        interpolation=cv2.INTER_LINEAR
    )
    
    # FPS
    fps = 1 / max(time.time() - prev_time, 1e-6)
    prev_time = time.time()
    
    # Проверяем успех
    is_success = info[0].get('is_success', False)
    
    # Добавляем информацию на кадр
    cv2.putText(frame, f"Episode: {episode_count + 1}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Step: {step}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Reward: {episode_reward:.2f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if episode_reward >= 0 else (0, 0, 255), 2)
    cv2.putText(frame, f"Success Rate: {success_count}/{episode_count}" if episode_count > 0 else "Success Rate: -", 
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
    
    if is_success:
        cv2.putText(frame, "SUCCESS!", (frame.shape[1]//2 - 100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    cv2.imshow("PandaGym - Best Model", frame)
    
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
        print(f"Episode {episode_count}: Reward = {episode_reward:.2f}, Success = {is_success}")
        obs = test_env.reset()
        episode_reward = 0
        step = 0

test_env.close()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print(f"Финальная статистика:")
print(f"Всего эпизодов: {episode_count}")
print(f"Успешных: {success_count}")
print(f"Success Rate: {success_count/episode_count*100:.1f}%" if episode_count > 0 else "N/A")
print("=" * 50)