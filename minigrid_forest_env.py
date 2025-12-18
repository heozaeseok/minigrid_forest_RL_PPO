import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Ball, Key
from minigrid.minigrid_env import MiniGridEnv
import random

# --- Custom Objects ---
class HealthyTree(Box):
    def __init__(self): super().__init__(color='green')
    def can_overlap(self): return True

class BurningTree(Ball):
    def __init__(self): super().__init__(color='red')
    def can_overlap(self): return True

class BurntTree(Box):
    def __init__(self): super().__init__(color='grey')
    def can_overlap(self): return True

class ExtinguishedTree(Box):
    def __init__(self): super().__init__(color='blue')
    def can_overlap(self): return True

class WaterTank(Key):
    def __init__(self): super().__init__(color='blue')
    def can_overlap(self): return True

class Stone(Box):
    def __init__(self): super().__init__(color='purple') 
    def can_overlap(self): return True 

# --- Environment ---
class ForestFireEnv(MiniGridEnv):
    def __init__(self, size=24, max_steps=1000, render_mode=None, 
                 fire_spread_prob=0.01,
                 burn_out_prob=0.001,
                 
                 # [보상 설정]
                 reward_extinguish_base=2.0,   
                 reward_risk_factor=3.0,       
                 # [수정] 고정 완료 보상 대신 나무 1그루당 생존 보상 설정
                 reward_per_surviving_tree=5.0, 
                 
                 penalty_step=-0.01,           
                 penalty_wall=-0.1,            
                 penalty_spread=-1.0,          
                 penalty_burnt=-0.5,
                 penalty_failure=-100.0
                 ):
        
        self.size = size
        self.base_spread_prob = fire_spread_prob
        self.burn_out_prob = burn_out_prob
        
        # 보상 변수 매핑
        self.r_ext_base = reward_extinguish_base
        self.r_risk_factor = reward_risk_factor
        self.r_per_tree = reward_per_surviving_tree # 변수 저장
        
        self.p_step = penalty_step
        self.p_wall = penalty_wall
        self.p_spread = penalty_spread
        self.p_burnt = penalty_burnt
        self.p_failure = penalty_failure
        
        self.max_water = 2
        self.current_water = 2
        self.tank_pos = (1, 1)
        self.steps_since_tank = 0
        self.initial_fire_count = 3
        
        self.fixed_stone_coords = [
            (15, 18), (16, 18), (15, 19), (16, 19),
            (18, 15), (19, 15), (18, 16), (19, 16),
            (13, 15), (14, 15), (13, 16), (14, 16)
        ]

        all_tree_coords = self._generate_organic_forest()
        self.fixed_tree_coords = [c for c in all_tree_coords if c not in self.fixed_stone_coords]

        mission_space = MissionSpace(mission_func=lambda: "Prioritize high risk fire")
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            render_mode=render_mode,
            see_through_walls=True
        )

        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

    def get_frame(self, highlight=False, tile_size=32, agent_pov=False):
        return super().get_frame(highlight=False, tile_size=tile_size, agent_pov=agent_pov)

    def _generate_organic_forest(self):
        coords = set()
        for x in range(self.size):
            for y in range(self.size):
                if ((x - 11) / 7)**2 + ((y - 19) / 3.5)**2 <= 1: coords.add((x, y))
        for x in range(self.size):
            for y in range(self.size):
                if ((x - 18) / 3.5)**2 + ((y - 11) / 7)**2 <= 1: coords.add((x, y))
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
        for (tx, ty) in self.fixed_tree_coords:
            self.grid.set(tx, ty, HealthyTree())
            self.trees.append((tx, ty))

        if len(self.trees) >= self.initial_fire_count:
            fire_indices = random.sample(range(len(self.trees)), self.initial_fire_count)
            for idx in fire_indices:
                fx, fy = self.trees[idx]
                self.grid.set(fx, fy, BurningTree())

        self.agent_pos = self.tank_pos
        self.agent_dir = 0
        self.current_water = self.max_water
        self.steps_since_tank = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.steps_since_tank = 0
        obs = self.gen_obs()
        return obs, info

    def _spread_fire_logic(self):
        spread_penalty = 0.0
        burnt_penalty = 0.0
        
        fire_locs = [pos for pos in self.trees if 
                     self.grid.get(*pos) and isinstance(self.grid.get(*pos), BurningTree)]
        
        for (fx, fy) in fire_locs:
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = fx+dx, fy+dy
                neighbor = self.grid.get(nx, ny)
                if neighbor and isinstance(neighbor, HealthyTree):
                    if random.random() < self.base_spread_prob:
                        self.grid.set(nx, ny, BurningTree())
                        spread_penalty += self.p_spread 
        
        for (fx, fy) in fire_locs:
            if random.random() < self.burn_out_prob:
                self.grid.set(fx, fy, BurntTree())
                burnt_penalty += self.p_burnt
                
        return spread_penalty + burnt_penalty

    def _count_fires(self):
        count = 0
        for pos in self.trees:
            cell = self.grid.get(*pos)
            if cell and isinstance(cell, BurningTree):
                count += 1
        return count

    def _count_healthy(self):
        count = 0
        for pos in self.trees:
            cell = self.grid.get(*pos)
            if cell and isinstance(cell, HealthyTree):
                count += 1
        return count

    def _get_risk_score(self, x, y):
        neighbor_trees = 0
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                cell = self.grid.get(nx, ny)
                if cell and isinstance(cell, HealthyTree):
                    neighbor_trees += 1
        return float(neighbor_trees)

    def _get_zone_health(self):
        zones = {'A': [], 'B': [], 'C': []}
        for (tx, ty) in self.trees:
            if tx < 14 and ty > 14: zones['A'].append((tx, ty))
            elif tx >= 14 and ty >= 14: zones['B'].append((tx, ty))
            else: zones['C'].append((tx, ty))
            
        ratios = []
        for z in ['A', 'B', 'C']:
            total = len(zones[z])
            if total == 0: 
                ratios.append(1.0)
                continue
            healthy = 0
            for (x, y) in zones[z]:
                cell = self.grid.get(x, y)
                if cell and isinstance(cell, HealthyTree):
                    healthy += 1
            ratios.append(healthy / total)
        return ratios

    def gen_obs(self):
        obs = np.zeros(10, dtype=np.float32)
        ax, ay = self.agent_pos
        
        obs[0] = ax / self.size
        obs[1] = ay / self.size
        obs[2] = self.current_water / self.max_water
        
        nearest_fire = None
        min_dist = 9999
        highest_risk_fire = None
        max_risk_score = -1.0
        
        for t_pos in self.trees:
            cell = self.grid.get(*t_pos)
            if cell and isinstance(cell, BurningTree):
                dist = abs(ax - t_pos[0]) + abs(ay - t_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_fire = t_pos
                
                risk = self._get_risk_score(*t_pos)
                if risk > max_risk_score:
                    max_risk_score = risk
                    highest_risk_fire = t_pos

        if nearest_fire:
            obs[3] = (nearest_fire[0] - ax) / self.size
            obs[4] = (nearest_fire[1] - ay) / self.size
            
        if highest_risk_fire:
            obs[5] = (highest_risk_fire[0] - ax) / self.size
            obs[6] = (highest_risk_fire[1] - ay) / self.size
            
        z_ratios = self._get_zone_health()
        obs[7] = z_ratios[0]
        obs[8] = z_ratios[1]
        obs[9] = z_ratios[2]
        return obs

    def step(self, action):
        self.step_count += 1        
        self.steps_since_tank += 1
        reward = self.p_step 
        terminated = False
        truncated = False
        
        if self.current_water == 0 or self.steps_since_tank >= 50:
            tx, ty = self.tank_pos
            ax, ay = self.agent_pos
            if ax < tx: action = 2
            elif ax > tx: action = 4
            elif ay < ty: action = 3
            elif ay > ty: action = 1
            else: action = 0 
        
        dx, dy = 0, 0
        if action == 1: dy = -1
        elif action == 2: dx = 1
        elif action == 3: dy = 1
        elif action == 4: dx = -1
        
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        
        if 0 <= nx < self.size and 0 <= ny < self.size:
            cell_at_dest = self.grid.get(nx, ny)
            if cell_at_dest and isinstance(cell_at_dest, BurningTree):
                if self.current_water > 0:
                    self.current_water -= 1
                    self.grid.set(nx, ny, ExtinguishedTree())
                    self.agent_pos = (nx, ny)
                    
                    risk_score = self._get_risk_score(nx, ny)
                    reward += self.r_ext_base + (risk_score * self.r_risk_factor)
                else:
                    self.agent_pos = (nx, ny)
            elif cell_at_dest and cell_at_dest.type == 'wall':
                reward += self.p_wall 
            else:
                self.agent_pos = (nx, ny)
        else:
            reward += self.p_wall 

        current_cell = self.grid.get(*self.agent_pos)
        if current_cell and isinstance(current_cell, WaterTank):
            if self.current_water < self.max_water:
                self.current_water = self.max_water
            self.steps_since_tank = 0
        
        reward += self._spread_fire_logic()

        fire_count = self._count_fires()
        healthy_count = self._count_healthy()

        # 1. 실패: 전멸
        if healthy_count == 0:
            reward = self.p_failure
            terminated = True
            
        # 2. 성공: 불 모두 진압
        elif fire_count == 0:
            # [수정] 성공 보상 = (남은 건강한 나무 수) * (설정된 계수, 기본 5)
            # 예: 나무가 100그루 남았다면 +500점
            reward += healthy_count * self.r_per_tree
            terminated = True
        
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}

# 환경 ID 등록

gym.register(id="ForestFireMLP-v22", entry_point=ForestFireEnv)
