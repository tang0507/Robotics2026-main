#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import contextlib
from pathlib import Path
import json

from ament_index_python.packages import get_package_share_directory
import rclpy
import trimesh
import speech_recognition as sr

from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
from myrobot_interfaces.srv import SetHanoiTowerStations
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from std_msgs.msg import Header

# 引入 Azure AI 相關套件
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

"""Variable for end-effector"""
EefState = 0

"""Hanoi tower geometry"""
Tower_base = 0.0014  
Tower_height = 0.025  
Tower_overlap = 0.015  

"""Hanoi tower position"""
STATION_POSITIONS = (
    (0.25, 0.15),
    (0.25, 0.0),
    (0.25, -0.15),
)
NUM_DISKS = 3
HANOI_TOWER_NAMES = tuple(f"tower{index}" for index in range(1, NUM_DISKS + 1))

"""Box geometry and position"""
BOX_SIZE = (0.1, 0.001, 0.1)
BOX_POSITIONS = {
    "left": (0.25, 0.075, 0.05),   # y = 0.075 偏向站點 0 (左)
    "right": (0.25, -0.075, 0.05), # y = -0.075 偏向站點 2 (右)
}

"""Hanoi tower mesh file path"""
MESH_DIR = Path(get_package_share_directory("myplan")) / "mesh"
MESH_FILE_PATH = {
    tower_name: str(MESH_DIR / f"{tower_name}.stl")
    for tower_name in HANOI_TOWER_NAMES
}
for mesh in MESH_FILE_PATH.values():
    if not Path(mesh).exists():
        raise FileNotFoundError(f"Mesh path error: {mesh}")
MESH_SCALE = (0.00095, 0.00095, 0.00095)

# ==================== GPT 提示詞設定 ====================
SYSTEM_PROMPT = """You are a helpful ROS 2 assistant for Hanoi Tower robot arm.
Given a user voice request, your job is to extract:
1. The target station index (0, 1, or 2) where the whole tower should move to.
2. The obstacle box configuration ("both", "left", "right", or "none").

Output ONLY plain JSON with keys "target_station" and "obstacles".

Example 1: "Move to station two with left obstacle only" -> {"target_station": 2, "obstacles": "left"}
Example 2: "幫我移到第一個站點，不需要障礙物" -> {"target_station": 0, "obstacles": "none"}
Example 3: "站點一，左右都要有障礙物" -> {"target_station": 1, "obstacles": "both"}
Example 4: "移到站點二，右邊放障礙物" -> {"target_station": 2, "obstacles": "right"}

Output JSON only. No markdown.
"""

def load_mesh_from_file(file_path: str, scale: tuple[float, float, float]) -> Mesh:
    mesh_data = trimesh.load(file_path, force="mesh")
    assert isinstance(mesh_data, trimesh.base.Trimesh)

    vertices = [
        Point(x=float(v[0]) * scale[0], y=float(v[1]) * scale[1], z=float(v[2]) * scale[2])
        for v in mesh_data.vertices
    ]
    triangles = [
        MeshTriangle(vertex_indices=[int(f[0]), int(f[1]), int(f[2])])
        for f in mesh_data.faces if len(f) == 3
    ]
    return Mesh(triangles=triangles, vertices=vertices)

def read_station(prompt: str) -> int:
    while True:
        try:
            station = int(input(prompt))
        except ValueError:
            print("請輸入整數站點索引：0, 1 或 2。")
            continue
        if 0 <= station < len(STATION_POSITIONS):
            return station
        print("站點索引必須是 0, 1 或 2。")

def build_stacks_from_tower_stations(tower_stations: tuple[int, ...]) -> list[list[str]]:
    stacks: list[list[str]] = [[] for _ in STATION_POSITIONS]
    for tower_name, station in zip(HANOI_TOWER_NAMES, tower_stations):
        stacks[station].append(tower_name)
    return stacks

@contextlib.contextmanager
def silence_native_stderr():
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


class MoveGroupPythonInterface(Node):
    def __init__(self, executor: SingleThreadedExecutor):
        super().__init__("hanoi_spawn_objects")
        self.PLANNING_FRAME = "world"
        self._executor = executor

        self.collision_object_publisher = self.create_publisher(CollisionObject, "/collision_object", 10)
        self.attached_collision_object_publisher = self.create_publisher(AttachedCollisionObject, "/attached_collision_object", 10)
        self.hanoi_station_client = self.create_client(SetHanoiTowerStations, "/set_hanoi_tower_stations")

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            raise RuntimeError("環境變數 GITHUB_TOKEN 未設定。請先 export GITHUB_TOKEN=your_token")
        
        self.client = ChatCompletionsClient(
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(token),
        )
        self.model = "gpt-4o"

        time.sleep(1.0)

    def wait_for_state_update(self) -> None:
        self._executor.spin_once(timeout_sec=0.1)

    # 新增：自動清除指定 ID 物件的功能
    def remove_object(self, object_name: str) -> None:
        collision_object = CollisionObject(
            header=Header(frame_id=self.PLANNING_FRAME, stamp=self.get_clock().now().to_msg()),
            id=object_name,
            operation=CollisionObject.REMOVE,
        )
        self.collision_object_publisher.publish(collision_object)
        self.wait_for_state_update()

    # 新增：一鍵清理所有可能的舊障礙物與河內塔物件
    def clear_all_environment_objects(self) -> None:
        self.get_logger().info("🧹 正在清理舊的環境物件與障礙物...")
        # 清除河內塔
        for tower_name in HANOI_TOWER_NAMES:
            self.remove_object(tower_name)
        # 清除障礙物
        self.remove_object("box_left")
        self.remove_object("box_right")
        # 額外確保等待環境同步
        time.sleep(0.5)

    def add_box(self, *, box_name: str, box_pose: Pose, size: tuple[float, float, float]) -> None:
        box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=size)
        collision_object = CollisionObject(
            header=Header(frame_id=self.PLANNING_FRAME, stamp=self.get_clock().now().to_msg()),
            id=box_name, primitives=[box], primitive_poses=[box_pose], operation=CollisionObject.ADD,
        )
        self.collision_object_publisher.publish(collision_object)
        self.get_logger().info(f"Added box: {box_name}")
        self.wait_for_state_update()

    def add_mesh(self, *, mesh_name: str, mesh_position: Point, file_path: str, scale: tuple[float, float, float]) -> None:
        pose = Pose(position=mesh_position, orientation=Quaternion(x=0.7071081, y=0.0, z=0.0, w=0.7071081))
        collision_object = CollisionObject(
            header=Header(frame_id=self.PLANNING_FRAME, stamp=self.get_clock().now().to_msg()),
            id=mesh_name, meshes=[load_mesh_from_file(file_path, scale)], mesh_poses=[pose], operation=CollisionObject.ADD,
        )
        self.collision_object_publisher.publish(collision_object)
        self.get_logger().info(f"Added mesh: {mesh_name}")
        self.wait_for_state_update()

    def get_target_and_obstacles_via_voice(self) -> tuple[int, str]:
        mic = sr.Microphone()

        while True:
            try:
                with silence_native_stderr(), mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                    print("\n🎙️ [語音輸入] 請說明目標站點與障礙物設定...")
                    print("   (例如：'移到站點二，只要左邊障礙物' 或 '站點零，不要障礙物')")
                    audio = self.recognizer.listen(source, timeout=10.0, phrase_time_limit=10.0)
                
                print("正在進行語音轉文字...")
                text = self.recognizer.recognize_google(audio, language="zh-TW")
                print(f"🗣️ 識別結果: {text}")

                print("正在請求 GPT 分析指令...")
                response = self.client.complete(
                    messages=[SystemMessage(SYSTEM_PROMPT), UserMessage(text)],
                    model=self.model,
                )
                gpt_reply = response.choices[0].message.content.strip()
                print(f"🤖 GPT 回應: {gpt_reply}")

                data = json.loads(gpt_reply)
                target_station = int(data["target_station"])
                obstacle_mode = str(data["obstacles"]).lower()
                
                valid_modes = ["both", "left", "right", "none"]
                if (0 <= target_station <= 2) and (obstacle_mode in valid_modes):
                    return target_station, obstacle_mode
                else:
                    print("❌ GPT 解析出的內容不合規範，請再試一次。")

            except sr.WaitTimeoutError:
                print("⏳ 聆聽逾時，沒有聽到聲音，請再試一次。")
            except sr.UnknownValueError:
                print("❓ 語音無法辨識，請講得更清晰一些。")
            except Exception as e:
                print(f"💥 發生錯誤: {e}，請再試一次。")

    def send_hanoi_station_request(self, tower_stations: tuple[int, ...], target_station: int) -> bool:
        self.get_logger().info("Waiting for /set_hanoi_tower_stations service...")
        if not self.hanoi_station_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("/set_hanoi_tower_stations service is not available")
            return False

        request = SetHanoiTowerStations.Request()
        request.tower_stations = list(tower_stations)
        request.target_station = target_station

        future = self.hanoi_station_client.call_async(request)
        while rclpy.ok() and not future.done():
            self._executor.spin_once(timeout_sec=0.1)

        if future.result() is None:
            return False
        response = future.result()
        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().error(response.message)
        return response.success


def main(args=None):
    # 1. 手動輸入 3 個 tower 的初始位置
    print("=== Step 1: 手動設定河內塔初始位置 ===")
    for index, (x, y) in enumerate(STATION_POSITIONS):
        print(f"  站點 {index}: x={x:.3f}, y={y:.3f}")

    tower_stations = tuple(
        read_station(f"請問 {tower_name} 在哪個站點？(0, 1, 2): ")
        for tower_name in HANOI_TOWER_NAMES
    )

    rclpy.init(args=args)
    path_plan_object = None

    try:
        executor = SingleThreadedExecutor()
        path_plan_object = MoveGroupPythonInterface(executor)
        executor.add_node(path_plan_object)

        # 【核心修改 1】在做任何事之前，先清除上一輪殘留的物件，確保環境乾淨，此時 RViz 什麼障礙物都沒有
        path_plan_object.clear_all_environment_objects()

        # 2. 語音輸入階段：此時 RViz 還不會顯示任何障礙物
        print("\n=== Step 2: 語音設定最終目標與障礙物 ===")
        target_station, obstacle_mode = path_plan_object.get_target_and_obstacles_via_voice()
        print(f" 🎯 成功取得目標！全塔將移動至站點: {target_station}")
        print(f" 🚧 障礙物模式設定為: {obstacle_mode}")

        # 【核心修改 2】語音輸入完畢後，才「同時」生成河內塔與選定的障礙物
        print("\n=== Step 3: 依據語音指令生成環境物件 ===")
        
        # 生成河內塔物體
        stacks = build_stacks_from_tower_stations(tower_stations)
        tower_spacing = Tower_height - Tower_overlap
        for station_index, stack in enumerate(stacks):
            station_x, station_y = STATION_POSITIONS[station_index]
            for stack_index, tower_name in enumerate(stack):
                tower_position = Point(
                    x=station_x, y=station_y, z=Tower_base + stack_index * tower_spacing,
                )
                path_plan_object.add_mesh(
                    mesh_name=tower_name, mesh_position=tower_position,
                    file_path=MESH_FILE_PATH[tower_name], scale=MESH_SCALE,
                )

        # 動態生成障礙物 Box
        if obstacle_mode in ["left", "both"]:
            x, y, z = BOX_POSITIONS["left"]
            path_plan_object.add_box(
                box_name="box_left",
                box_pose=Pose(orientation=Quaternion(w=1.0), position=Point(x=x, y=y, z=z)),
                size=BOX_SIZE,
            )
        
        if obstacle_mode in ["right", "both"]:
            x, y, z = BOX_POSITIONS["right"]
            path_plan_object.add_box(
                box_name="box_right",
                box_pose=Pose(orientation=Quaternion(w=1.0), position=Point(x=x, y=y, z=z)),
                size=BOX_SIZE,
            )
            
        if obstacle_mode == "none":
            print(" 🚫 語音指示不產生任何障礙物。")

        # 3. 發送服務給手臂 Planner 開始動作
        path_plan_object.send_hanoi_station_request(tower_stations, target_station)

        print("\n任務已發送！手臂規劃中...")
        input("按下 Enter 鍵結束程式並退出環境...")

    except KeyboardInterrupt:
        print("使用者中斷程式")
    finally:
        if path_plan_object is not None:
            path_plan_object.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()