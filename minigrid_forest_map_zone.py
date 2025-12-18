import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Ball, Key
from minigrid.minigrid_env import MiniGridEnv
import random
import time

# --- Custom Objects ---
# 구역별 나무 색상 다르게 설정
class TreeZoneA(Box):
    def __init__(self): super().__init__(color='green') # A구역: 초록
    def can_overlap(self): return True

class TreeZoneB(Box):
    def __init__(self): super().__init__(color='yellow') # B구역: 노랑
    def can_overlap(self): return True

class TreeZoneC(Box):
    def __init__(self): super().__init__(color='purple') # C구역: 보라
    def can_overlap(self): return True

class BurningTree(Ball):
    def __init__(self): super().__init__(color='red')
    def can_overlap(self): return True

class WaterTank(Key):
    def __init__(self): super().__init__(color='blue')
    def can_overlap(self): return True 

class Stone(Box):
    def __init__(self): 
        # C구역(보라)과 겹치지 않게 파란색으로 변경 (모양은 Box라 Tank(Key)와 구분됨)
        super().__init__(color='blue') 
    def can_overlap(self): 
        return True 

# --- Visualization Environment ---
class ForestFireMapViewer(MiniGridEnv):
    def __init__(self, size=24, render_mode="human"):
        self.size = size
        self.tank_pos = (1, 1)
        
        self.fixed_stone_coords = [
            (15, 18), (16, 18), (15, 19), (16, 19),
            (18, 15), (19, 15), (18, 16), (19, 16),
            (13, 15), (14, 15), (13, 16), (14, 16)
        ]

        all_tree_coords = self._generate_organic_forest()
        self.fixed_tree_coords = [c for c in all_tree_coords if c not in self.fixed_stone_coords]

        mission_space = MissionSpace(mission_func=lambda: "Zone Visualization: A(Grn) B(Yel) C(Pur)")
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=100,
            render_mode=render_mode,
            see_through_walls=True
        )
        self.highlight = False 

    def _generate_organic_forest(self):
        coords = set()
        # 1. 아래쪽 넓은 영역
        for x in range(self.size):
            for y in range(self.size):
                if ((x - 11) / 7)**2 + ((y - 19) / 3.5)**2 <= 1: coords.add((x, y))
        # 2. 오른쪽 위로 뻗는 영역
        for x in range(self.size):
            for y in range(self.size):
                if ((x - 18) / 3.5)**2 + ((y - 11) / 7)**2 <= 1: coords.add((x, y))
        # 3. 연결부 보정
        for x in range(self.size):
            for y in range(self.size):
                if ((x - 16) / 3.5)**2 + ((y - 16) / 3.5)**2 <= 1: coords.add((x, y))

        valid_coords = []
        for (x, y) in coords:
            if 2 <= x < self.size - 2 and 2 <= y < self.size - 2:
                if (x, y) != self.tank_pos:
                    valid_coords.append((x, y))
        return valid_coords

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(WaterTank(), *self.tank_pos)
        
        for (sx, sy) in self.fixed_stone_coords:
            self.grid.set(sx, sy, Stone())

        self.trees = []
        
        # [핵심] 좌표에 따라 다른 색상의 나무 배치
        for (tx, ty) in self.fixed_tree_coords:
            # 구역 판별 로직 (학습 환경과 동일)
            # Zone A: 왼쪽 꼬리 (x < 14, y > 14)
            # Zone B: 오른쪽 꺾임 (나머지 중 y > 14) -> 즉 x >= 14
            # Zone C: 위쪽 머리 (y <= 14)
            
            if ty <= 14: # y좌표가 위쪽이면 Zone C
                tree_obj = TreeZoneC()
            elif tx < 14: # 아래쪽이면서 왼쪽이면 Zone A
                tree_obj = TreeZoneA()
            else: # 나머지(아래쪽이면서 오른쪽) Zone B
                tree_obj = TreeZoneB()

            self.grid.set(tx, ty, tree_obj)
            self.trees.append((tx, ty))

        # 화재는 빨간색 공으로 표시 (기존 유지)
        if len(self.trees) >= 3:
            fire_indices = random.sample(range(len(self.trees)), 3)
            for idx in fire_indices:
                fx, fy = self.trees[idx]
                self.grid.set(fx, fy, BurningTree())

        self.agent_pos = self.tank_pos
        self.agent_dir = 0

if __name__ == "__main__":
    env = ForestFireMapViewer(render_mode="human")
    
    print("Map Visualization with Zones:")
    print(" - Zone A (Bottom Left): Green Box")
    print(" - Zone B (Bottom Right): Yellow Box")
    print(" - Zone C (Top Head): Purple Box")
    print(" - Stones: Blue Box")
    print(" - Fire: Red Ball")
    
    env.reset()
    env.render()
    
    # 창을 좀 더 오래 띄워둠
    time.sleep(100) 
    env.close()