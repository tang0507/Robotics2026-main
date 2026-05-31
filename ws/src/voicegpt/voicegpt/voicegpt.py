#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading
import contextlib
import concurrent.futures
import speech_recognition as sr

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# # ===============================================================
# # TODO: 你可以修改 SYSTEM_PROMPT 來調整 GPT 的行為。
# SYSTEM_PROMPT = """You are a helpful ROS 2 assistant for controlling turtlesim.
# Given a user request (e.g., 'move forward 2m with speed 1m/s'), output ONLY plain JSON (no markdown).
# You may output either:
# (1) a single command object, or
# (2) a list of command objects to be executed sequentially.

# Each command object must follow:
# {
#   "action": "move",
#   "linear_x": <float>,
#   "distance": <float>,
# }

# Notes:
# - Always output exactly these 3 keys: action, linear_x, distance.
# - For action="move": set linear_x/distance to valid values that user gives, if user didn't mention them, set linear_x=0.5 and distance=1.0.
# - linear_x: forward > 0, backward < 0
# - distance: meters


# Output JSON only.
# """
# # ===============================================================原本程式

# ===============================================================
# TODO: 你可以修改 SYSTEM_PROMPT 來調整 GPT 的行為。
SYSTEM_PROMPT = """You are a helpful ROS 2 assistant for controlling turtlesim.
Given a user request (e.g., 'move forward 2m' or 'rotate left 90 degrees'), output ONLY plain JSON (no markdown).
You may output either:
(1) a single command object, or
(2) a list of command objects to be executed sequentially.

Each command object must follow one of these structures:

1. For moving:
{
  "action": "move",
  "linear_x": <float>,
  "distance": <float>
}

2. For rotating:
{
  "action": "rotate",
  "angular_z": <float>,
  "rotate_duration": <float>
}

Notes:
- For action="move": linear_x (m/s), distance (meters). Forward > 0, backward < 0.
- For action="rotate": angular_z (rad/s), rotate_duration (seconds). 
  * Counter-clockwise (left) > 0, Clockwise (right) < 0.
  * 90 degrees is approximately 1.57 radians.
- If the user doesn't specify speed/duration, use reasonable defaults (e.g., speed=0.5, duration=1.0).

Output JSON only.
"""
# ===============================================================

class VoiceGPTNode(Node):
    def __init__(self):
        super().__init__('voice_gpt_node')

        # ROS2 Publisher
        self.pub = self.create_publisher(String, 'gpt_reply_to_user', 10)

        # Speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = float(os.environ.get("VOICEGPT_PAUSE_THRESHOLD", "1.0"))
        self.recognizer.non_speaking_duration = float(
            os.environ.get("VOICEGPT_NON_SPEAKING_DURATION", "0.6")
        )
        self.recognizer.phrase_threshold = float(os.environ.get("VOICEGPT_PHRASE_THRESHOLD", "0.3"))
        self.listen_timeout_sec = float(os.environ.get("VOICEGPT_LISTEN_TIMEOUT", "5"))
        self.phrase_time_limit_sec = float(os.environ.get("VOICEGPT_PHRASE_TIME_LIMIT", "15"))
        self.ambient_adjust_sec = float(os.environ.get("VOICEGPT_AMBIENT_ADJUST_SEC", "1.0"))
        # Optional mic index: use default device when unset.
        self.mic_device_index = self._parse_mic_index(os.environ.get("VOICEGPT_MIC_DEVICE"))

        # GitHub Models (Azure AI Inference) client
        endpoint = "https://models.github.ai/inference"
        model = "openai/gpt-4.1"
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise RuntimeError("環境變數 GITHUB_TOKEN 未設定。請先 export GITHUB_TOKEN=your token")

        self.model = model
        self.gpt_timeout_sec = 20
        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
        )

        # Start speech thread
        self._thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._thread.start()

        self.get_logger().info("VoiceGPTNode started. Listening from microphone...")

    def _call_gpt(self, user_text: str) -> str:
        response = self.client.complete(
            messages=[
                SystemMessage(SYSTEM_PROMPT),
                UserMessage(user_text),
            ],
            model=self.model,
        )
        return response.choices[0].message.content

    def _call_gpt_with_timeout(self, user_text: str) -> str:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(self._call_gpt, user_text)
        try:
            return future.result(timeout=self.gpt_timeout_sec)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise
        finally:
            # Do not block on hung network calls.
            pool.shutdown(wait=False, cancel_futures=True)

    def _speech_loop(self):
        mic = sr.Microphone(device_index=self.mic_device_index)

        while rclpy.ok():
            try:
                with self._silence_native_stderr(), mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=self.ambient_adjust_sec)
                    self.get_logger().info("請開始說話...")
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.listen_timeout_sec,
                        phrase_time_limit=self.phrase_time_limit_sec,
                    )
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                self.get_logger().error(f"Microphone/listen error: {e}")
                continue

            try:
                text = self.recognizer.recognize_google(audio, language="zh-CN")
                self.get_logger().info(f"Recognized: {text}")

                self.get_logger().info("Calling GPT...")
                gpt_reply = self._call_gpt_with_timeout(text)
                self.get_logger().info(f"GPT reply: {gpt_reply}")

                msg = String()
                msg.data = gpt_reply
                self.pub.publish(msg)

            except sr.UnknownValueError:
                self.get_logger().warning("語音無法辨識")
            except sr.RequestError as e:
                self.get_logger().error(f"SpeechRecognition API error: {e}")
            except concurrent.futures.TimeoutError:
                self.get_logger().error(
                    f"GPT request timed out after {self.gpt_timeout_sec}s. Check network/token."
                )
            except Exception as e:
                self.get_logger().error(f"GPT call/publish error: {e}")

    def _parse_mic_index(self, raw_value: str):
        if not raw_value:
            return None
        try:
            return int(raw_value)
        except ValueError:
            self.get_logger().warning(
                f"Invalid VOICEGPT_MIC_DEVICE='{raw_value}', fallback to default microphone."
            )
            return None

    @contextlib.contextmanager
    def _silence_native_stderr(self):
        # ALSA/PortAudio writes directly to fd=2; redirect fd to /dev/null in this block.
        stderr_fd = 2
        saved_fd = os.dup(stderr_fd)
        try:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                os.dup2(devnull.fileno(), stderr_fd)
                yield
        finally:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)


def main():
    rclpy.init()
    node = VoiceGPTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
