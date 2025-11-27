import gymnasium as gym
import panda_gym
import numpy as np
import cv2
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os

ENV_NAME = "PandaReach-v3"
MAX_EPISODE_STEPS = 50
TARGET_RANGE = 0.3
WINDOW_SCALE = 1.5
MODEL_DIR = "./models/"
TARGET_FPS = 30  # Целевой FPS для плавности
SLOWDOWN = 3  # Во сколько раз замедлить (1 = обычная скорость, 2 = в 2 раза медленнее)

print("Загрузка модели...")
model = PPO.load(MODEL_DIR + "best_model")

def make_env():
    env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
    env.unwrapped.target_range = TARGET_RANGE
    return env

test_env = DummyVecEnv([make_env])

# Ищем файл нормализации
norm_files = ["vec_normalize_final.pkl", "vec_normalize_interrupted.pkl", "vec_normalize.pkl"]
norm_loaded = False
for norm_file in norm_files:
    norm_path = MODEL_DIR + norm_file
    if os.path.exists(norm_path):
        print(f"Загрузка нормализации из {norm_file}...")
        test_env = VecNormalize.load(norm_path, test_env)
        test_env.training = False
        test_env.norm_reward = False
        norm_loaded = True
        break

if not norm_loaded:
    print("⚠️ ВНИМАНИЕ: Нормализация не найдена!")

print(f"\nЗапуск визуализации (замедление x{SLOWDOWN}, нажми 'q' для выхода)...\n")

obs = test_env.reset()
episode_reward = 0
episode_count = 0
success_count = 0
step = 0
frame_time = 1.0 / TARGET_FPS
last_frame_time = time.time()

# Для интерполяции между кадрами
prev_frame = None
interpolation_steps = SLOWDOWN  # Количество промежуточных кадров

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    episode_reward += reward[0]
    step += 1
    
    # Получаем новый кадр
    current_frame = test_env.envs[0].render()
    current_frame = (current_frame * 255).astype(np.uint8)
    current_frame = cv2.resize(
        current_frame, 
        (int(current_frame.shape[1]*WINDOW_SCALE), int(current_frame.shape[0]*WINDOW_SCALE)),
        interpolation=cv2.INTER_LINEAR
    )
    
    # Если есть предыдущий кадр, делаем плавную интерполяцию
    if prev_frame is not None and interpolation_steps > 1:
        for i in range(interpolation_steps):
            # Плавное смешивание кадров
            alpha = i / interpolation_steps
            interpolated_frame = cv2.addWeighted(
                prev_frame, 1 - alpha,
                current_frame, alpha,
                0
            )
            
            display_frame = interpolated_frame.copy()
            is_success = info[0].get('is_success', False)
            
            # Добавляем текст
            cv2.putText(display_frame, f"Episode: {episode_count + 1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display_frame, f"Step: {step}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display_frame, f"Reward: {episode_reward:.2f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if episode_reward >= 0 else (0,0,255), 2)
            cv2.putText(display_frame, f"Success: {success_count}/{episode_count}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            if is_success:
                cv2.putText(display_frame, "SUCCESS!", (display_frame.shape[1]//2 - 100, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            
            cv2.imshow("Panda Reach - Smooth", display_frame)
            
            # Держим стабильный FPS
            elapsed = time.time() - last_frame_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
            last_frame_time = time.time()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                test_env.close()
                cv2.destroyAllWindows()
                exit()
    else:
        # Первый кадр - просто показываем
        display_frame = current_frame.copy()
        is_success = info[0].get('is_success', False)
        
        cv2.putText(display_frame, f"Episode: {episode_count + 1}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(display_frame, f"Step: {step}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(display_frame, f"Reward: {episode_reward:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if episode_reward >= 0 else (0,0,255), 2)
        cv2.putText(display_frame, f"Success: {success_count}/{episode_count}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        if is_success:
            cv2.putText(display_frame, "SUCCESS!", (display_frame.shape[1]//2 - 100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
        
        cv2.imshow("Panda Reach - Smooth", display_frame)
    
    # Сохраняем текущий кадр для следующей интерполяции
    prev_frame = current_frame.copy()
    
    if done[0]:
        if is_success:
            success_count += 1
        episode_count += 1
        print(f"Episode {episode_count}: Reward={episode_reward:.2f}, Success={is_success}, Steps={step}")
        time.sleep(1.0)  # Пауза между эпизодами
        obs = test_env.reset()
        episode_reward = 0
        step = 0
        prev_frame = None  # Сбрасываем для нового эпизода

test_env.close()
cv2.destroyAllWindows()
print(f"\n{success_count}/{episode_count} успехов ({success_count/episode_count*100:.1f}%)")