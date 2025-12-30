import os
import sys
import time
import json
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from nitrogen.game_env import GamepadEnv
from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO
from nitrogen.inference_viz import create_viz, VideoRecorder
from nitrogen.inference_client import ModelClient

import argparse
parser = argparse.ArgumentParser(description="VLM Inference")
parser.add_argument("--process", type=str, default="celeste.exe", help="Game to play")
parser.add_argument("--allow-menu", action="store_true", help="Allow menu actions (Disabled by default)")
parser.add_argument("--port", type=int, default=5555, help="Port for model server")
parser.add_argument("--save-debug", action="store_true", help="Save debug screenshots (reduces performance)")
parser.add_argument("--save-actions", action="store_true", help="Save action log (reduces performance)")
parser.add_argument("--game-speed", type=float, default=1.0, help="Game speed multiplier (0.5 = half speed, 2.0 = double speed)")
parser.add_argument("--env-fps", type=int, default=60, help="Environment FPS (lower = less frequent AI queries)")
parser.add_argument("--profile", action="store_true", help="Show timing information for performance profiling")

args = parser.parse_args()

policy = ModelClient(port=args.port)
policy.reset()
policy_info = policy.info()
action_downsample_ratio = policy_info["action_downsample_ratio"]

CKPT_NAME = Path(policy_info["ckpt_path"]).stem
NO_MENU = not args.allow_menu

PATH_DEBUG = PATH_REPO / "debug"
PATH_DEBUG.mkdir(parents=True, exist_ok=True)

PATH_OUT = (PATH_REPO / "out" / CKPT_NAME).resolve()
PATH_OUT.mkdir(parents=True, exist_ok=True)

BUTTON_PRESS_THRES = 0.5

# Only create file paths if saving is enabled
if args.save_actions:
    video_files = sorted(PATH_OUT.glob("*_DEBUG.mp4"))
    if video_files:
        existing_numbers = [f.name.split("_")[0] for f in video_files]
        existing_numbers = [int(n) for n in existing_numbers if n.isdigit()]
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1
    PATH_ACTIONS = PATH_OUT / f"{next_number:04d}_ACTIONS.json"

def preprocess_img(main_image):
    # Optimized: Direct resize without unnecessary color conversions
    if isinstance(main_image, Image.Image):
        return main_image.resize((256, 256), Image.BILINEAR)
    else:
        return Image.fromarray(main_image).resize((256, 256), Image.BILINEAR)

zero_action = OrderedDict(
        [ 
            ("WEST", 0),
            ("SOUTH", 0),
            ("BACK", 0),
            ("DPAD_DOWN", 0),
            ("DPAD_LEFT", 0),
            ("DPAD_RIGHT", 0),
            ("DPAD_UP", 0),
            ("GUIDE", 0),
            ("AXIS_LEFTX", np.array([0], dtype=np.long)),
            ("AXIS_LEFTY", np.array([0], dtype=np.long)),
            ("LEFT_SHOULDER", 0),
            ("LEFT_TRIGGER", np.array([0], dtype=np.long)),
            ("AXIS_RIGHTX", np.array([0], dtype=np.long)),
            ("AXIS_RIGHTY", np.array([0], dtype=np.long)),
            ("LEFT_THUMB", 0),
            ("RIGHT_THUMB", 0),
            ("RIGHT_SHOULDER", 0),
            ("RIGHT_TRIGGER", np.array([0], dtype=np.long)),
            ("START", 0),
            ("EAST", 0),
            ("NORTH", 0),
        ]
    )

TOKEN_SET = BUTTON_ACTION_TOKENS

print("Model loaded, starting environment...")
print(f"Game speed: {args.game_speed}x, Environment FPS: {args.env_fps}")
for i in range(3):
    print(f"{3 - i}...")
    time.sleep(1)

env = GamepadEnv(
    game=args.process,
    game_speed=args.game_speed,  # Configurable game speed
    env_fps=args.env_fps,        # Configurable FPS
    async_mode=True,              # Async mode enabled for better performance
)

# These games requires to open a menu to initialize the controller
if args.process == "isaac-ng.exe":
    print(f"GamepadEnv ready for {args.process} at {env.env_fps} FPS")
    input("Press enter to create a virtual controller and start rollouts...")
    for i in range(3):
        print(f"{3 - i}...")
        time.sleep(1)

    def press(button):
        env.gamepad_emulator.press_button(button)
        env.gamepad_emulator.gamepad.update()
        time.sleep(0.05)
        env.gamepad_emulator.release_button(button)
        env.gamepad_emulator.gamepad.update()

    press("SOUTH")
    for k in range(5):
        press("EAST")
        time.sleep(0.3)

if args.process == "Cuphead.exe":
    print(f"GamepadEnv ready for {args.process} at {env.env_fps} FPS")
    input("Press enter to create a virtual controller and start rollouts...")
    for i in range(3):
        print(f"{3 - i}...")
        time.sleep(1)

    def press(button):
        env.gamepad_emulator.press_button(button)
        env.gamepad_emulator.gamepad.update()
        time.sleep(0.05)
        env.gamepad_emulator.release_button(button)
        env.gamepad_emulator.gamepad.update()

    press("SOUTH")
    for k in range(5):
        press("EAST")
        time.sleep(0.3)

env.reset()
env.pause()

# Initial call to get state
obs, reward, terminated, truncated, info = env.step(action=zero_action)

frames = None
step_count = 0

print(f"Performance mode: Debug saving={'ON' if args.save_debug else 'OFF'}, Action logging={'ON' if args.save_actions else 'OFF'}, Profiling={'ON' if args.profile else 'OFF'}")

# Profiling variables
total_preprocess_time = 0
total_predict_time = 0
total_action_time = 0
profile_samples = 0

try:
    while True:
        loop_start = time.perf_counter()
        
        # Preprocess
        preprocess_start = time.perf_counter()
        obs = preprocess_img(obs)
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        
        # Only save screenshots if debug flag is enabled
        if args.save_debug:
            obs.save(PATH_DEBUG / f"{step_count:05d}.png")

        # Predict
        predict_start = time.perf_counter()
        pred = policy.predict(obs)
        predict_time = (time.perf_counter() - predict_start) * 1000

        j_left, j_right, buttons = pred["j_left"], pred["j_right"], pred["buttons"]

        n = len(buttons)
        assert n == len(j_left) == len(j_right), "Mismatch in action lengths"

        # Build actions
        action_start = time.perf_counter()
        env_actions = []

        for i in range(n):
            move_action = zero_action.copy()

            xl, yl = j_left[i]
            xr, yr = j_right[i]
            move_action["AXIS_LEFTX"] = np.array([int(xl * 32767)], dtype=np.long)
            move_action["AXIS_LEFTY"] = np.array([int(yl * 32767)], dtype=np.long)
            move_action["AXIS_RIGHTX"] = np.array([int(xr * 32767)], dtype=np.long)
            move_action["AXIS_RIGHTY"] = np.array([int(yr * 32767)], dtype=np.long)
            
            button_vector = buttons[i]
            assert len(button_vector) == len(TOKEN_SET), "Button vector length does not match token set length"
            
            for name, value in zip(TOKEN_SET, button_vector):
                if "TRIGGER" in name:
                    move_action[name] = np.array([value * 255], dtype=np.long)
                else:
                    move_action[name] = 1 if value > BUTTON_PRESS_THRES else 0

            env_actions.append(move_action)

        for i, a in enumerate(env_actions):
            if NO_MENU:
                if a["START"]:
                    print("Model predicted start, disabling this action")
                a["GUIDE"] = 0
                a["START"] = 0
                a["BACK"] = 0

            for _ in range(action_downsample_ratio):
                obs, reward, terminated, truncated, info = env.step(action=a)

        action_time = (time.perf_counter() - action_start) * 1000

        # Only save actions if logging flag is enabled
        if args.save_actions:
            with open(PATH_ACTIONS, "a") as f:
                for i, a in enumerate(env_actions):
                    # convert numpy arrays to lists for JSON serialization
                    for k, v in a.items():
                        if isinstance(v, np.ndarray):
                            a[k] = v.tolist()
                    a["step"] = step_count
                    a["substep"] = i
                    json.dump(a, f)
                    f.write("\n")

        loop_time = (time.perf_counter() - loop_start) * 1000
        
        # Update profiling stats
        if args.profile:
            total_preprocess_time += preprocess_time
            total_predict_time += predict_time
            total_action_time += action_time
            profile_samples += 1
            
            # Print stats every 30 frames
            if step_count % 30 == 0 and step_count > 0:
                avg_preprocess = total_preprocess_time / profile_samples
                avg_predict = total_predict_time / profile_samples
                avg_action = total_action_time / profile_samples
                avg_total = (avg_preprocess + avg_predict + avg_action)
                
                print(f"\n=== Performance Stats (last 30 frames) ===")
                print(f"Preprocess: {avg_preprocess:.1f}ms ({avg_preprocess/avg_total*100:.1f}%)")
                print(f"AI Predict: {avg_predict:.1f}ms ({avg_predict/avg_total*100:.1f}%)")
                print(f"Actions:    {avg_action:.1f}ms ({avg_action/avg_total*100:.1f}%)")
                print(f"Total:      {avg_total:.1f}ms ({1000/avg_total:.1f} FPS)")
                print(f"=========================================\n")
                
                # Reset counters
                total_preprocess_time = 0
                total_predict_time = 0
                total_action_time = 0
                profile_samples = 0
        else:
            # Simple output without profiling
            print(f"Step {step_count}: Total loop time: {loop_time:.1f}ms ({1000/loop_time:.1f} FPS)")

        step_count += 1
        
finally:
    env.unpause()
    env.close()