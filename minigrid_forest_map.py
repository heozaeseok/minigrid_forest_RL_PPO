import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Ball, Key
from minigrid.minigrid_env import MiniGridEnv
import random
import time

# --- Custom Objects ---
class HealthyTree(Box):
    def __init__(self): super().__init__(color='green')
    def can_overlap(self): return True 

class BurningTree(Ball):
    def __init__(self): super().__init__(color='red')
    def can_overlap(self): return True

class WaterTank(Key):
    def __init__(self): super().__init__(color='blue')
    def can_overlap(self): return True 

class Stone(Box):
    def __init__(self): 
        super().__init__(color='purple')
    def can_overlap(self): 
        return True 

# --- Visualization Environment ---
class ForestFireMapViewer(MiniGridEnv):
    def __init__(self, size=24, render_mode="human"):
        self.size = size
        self.tank_pos = (1, 1)
        
        # 바위 좌표 설정
        self.fixed_stone_coords = [
            (15, 18), (16, 18), (15, 19), (16, 19),
            (18, 15), (19, 15), (18, 16), (19, 16),
            (13, 15), (14, 15), (13, 16), (14, 16)
        ]

        # 숲 좌표 생성
        all_tree_coords = self._generate_organic_forest()
        # 바위 위치에는 나무 생성 제외
        self.fixed_tree_coords = [c for c in all_tree_coords if c not in self.fixed_stone_coords]

        mission_space = MissionSpace(mission_func=lambda: "Map Visualization Mode")
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=100,
            render_mode=render_mode,
            see_through_walls=True
        )
        
        # [수정] 시야 하이라이트(흰색 네모) 제거
        self.highlight = False 

    # 숲 모양 생성 로직
    def _generate_organic_forest(self):
        coords = set()
        # 1. 아래쪽 넓은 영역
        for x in range(self.size):
            for y in range(self.size):
                if ((x - 11) / 7)**2 + ((y - 19) / 3.5)**2 <= 1:
                    coords.add((x, y))
        # 2. 오른쪽 위로 뻗는 영역
        for x in range(self.size):
            for y in range(self.size):
                if ((x - 18) / 3.5)**2 + ((y - 11) / 7)**2 <= 1:
                    coords.add((x, y))
        # 3. 연결부 보정
        for x in range(self.size):
            for y in range(self.size):
                if ((x - 16) / 3.5)**2 + ((y - 16) / 3.5)**2 <= 1:
                     coords.add((x, y))

        valid_coords = []
        for (x, y) in coords:
            if 2 <= x < self.size - 2 and 2 <= y < self.size - 2:
                if (x, y) != self.tank_pos:
                    valid_coords.append((x, y))
        return valid_coords

    # 그리드 그리기
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(WaterTank(), *self.tank_pos) # 급수소 배치
        
        # 바위 배치
        for (sx, sy) in self.fixed_stone_coords:
            self.grid.set(sx, sy, Stone())

        self.trees = []
        # 나무 배치
        for (tx, ty) in self.fixed_tree_coords:
            self.grid.set(tx, ty, HealthyTree())
            self.trees.append((tx, ty))

        # 초기 화재 배치 (랜덤 3개)
        if len(self.trees) >= 3:
            fire_indices = random.sample(range(len(self.trees)), 3)
            for idx in fire_indices:
                fx, fy = self.trees[idx]
                self.grid.set(fx, fy, BurningTree())

        self.agent_pos = self.tank_pos
        self.agent_dir = 0

# --- Main Execution ---
if __name__ == "__main__":
    env = ForestFireMapViewer(render_mode="human")
    
    print("Rendering Map... (Window will close in 5 seconds)")
    env.reset()
    env.render()
    
    time.sleep(5)
    
    env.close()
    print("Closed.")