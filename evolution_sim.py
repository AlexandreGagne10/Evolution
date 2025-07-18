import argparse
import math
import pygame
import torch

WIDTH, HEIGHT = 800, 600
FOOD_PER_ROUND = 50
FOOD_RADIUS = 3
AGENT_RADIUS = 4
MAX_AGE = 30
SURVIVAL_FOOD = 3
REPRODUCTION_THRESHOLD = 5
REPRODUCTION_COST = 2
MOVE_SPEED = 2.0


class Simulation:
    def __init__(self, num_agents: int, device: str = "cpu", initial_energy: int = SURVIVAL_FOOD):
        self.device = torch.device(device)
        self.positions = torch.rand(num_agents, 2, device=self.device) * torch.tensor([WIDTH, HEIGHT], device=self.device)
        self.energy = torch.full((num_agents,), initial_energy, device=self.device)
        self.age = torch.zeros(num_agents, dtype=torch.int32, device=self.device)
        self.food = torch.empty(0, 2, device=self.device)

    def spawn_food(self):
        food_positions = torch.rand(FOOD_PER_ROUND, 2, device=self.device) * torch.tensor([WIDTH, HEIGHT], device=self.device)
        if self.food.numel() == 0:
            self.food = food_positions
        else:
            self.food = torch.cat([self.food, food_positions], dim=0)

    def step(self):
        angles = torch.rand(len(self.positions), device=self.device) * 2 * math.pi
        delta = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * MOVE_SPEED
        self.positions += delta
        self.positions.clamp_(torch.tensor([0, 0], device=self.device), torch.tensor([WIDTH, HEIGHT], device=self.device))

        if self.food.numel() > 0:
            diff = self.positions.unsqueeze(1) - self.food.unsqueeze(0)
            dist = diff.norm(dim=2)
            eaten = dist < FOOD_RADIUS
            if eaten.any():
                agent_has_food = eaten.any(dim=1)
                self.energy[agent_has_food] += eaten[agent_has_food].float().sum(dim=1)
                food_keep = ~eaten.any(dim=0)
                self.food = self.food[food_keep]

        reproduce_mask = self.energy >= REPRODUCTION_THRESHOLD
        reproduce_idx = reproduce_mask.nonzero(as_tuple=True)[0]
        reproduce_idx = reproduce_idx[torch.randperm(len(reproduce_idx))]
        pairs = reproduce_idx.reshape(-1, 2)
        new_agents = []
        for a, b in pairs:
            if self.energy[a] >= REPRODUCTION_THRESHOLD and self.energy[b] >= REPRODUCTION_THRESHOLD:
                pos = (self.positions[a] + self.positions[b]) / 2
                new_agents.append(pos)
                self.energy[a] -= REPRODUCTION_COST
                self.energy[b] -= REPRODUCTION_COST
        if new_agents:
            new_pos = torch.stack(new_agents)
            self.positions = torch.cat([self.positions, new_pos], dim=0)
            self.energy = torch.cat([self.energy, torch.zeros(len(new_agents), device=self.device)], dim=0)
            self.age = torch.cat([self.age, torch.zeros(len(new_agents), dtype=torch.int32, device=self.device)], dim=0)

        self.age += 1
        alive = (self.age < MAX_AGE) & (self.energy >= SURVIVAL_FOOD)
        self.energy[alive] -= SURVIVAL_FOOD
        self.positions = self.positions[alive]
        self.energy = self.energy[alive]
        self.age = self.age[alive]


class Slider:
    def __init__(self, rect, min_val=0.1, max_val=5.0, value=1.0):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.handle_radius = 8
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            ratio = (event.pos[0] - self.rect.x) / self.rect.width
            ratio = max(0.0, min(1.0, ratio))
            self.value = self.min_val + ratio * (self.max_val - self.min_val)

    def draw(self, surface):
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        cx = self.rect.x + ratio * self.rect.width
        cy = self.rect.centery
        pygame.draw.circle(surface, (255, 0, 0), (int(cx), int(cy)), self.handle_radius)


def main():
    parser = argparse.ArgumentParser(description="Evolution Simulation")
    parser.add_argument("--agents", type=int, default=100, help="Number of starting agents")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT + 40))
    pygame.display.set_caption("Evolution Simulation")
    clock = pygame.time.Clock()

    sim = Simulation(args.agents, device=args.device)
    slider = Slider((10, HEIGHT + 5, WIDTH - 20, 20))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            slider.handle_event(event)

        speed = slider.value
        clock.tick(60)
        for _ in range(int(speed)):
            sim.spawn_food()
            sim.step()

        screen.fill((0, 0, 0))
        if sim.food.numel() > 0:
            for f in sim.food.cpu().tolist():
                pygame.draw.circle(screen, (0, 255, 0), (int(f[0]), int(f[1])), FOOD_RADIUS)
        for p in sim.positions.cpu().tolist():
            pygame.draw.circle(screen, (255, 255, 255), (int(p[0]), int(p[1])), AGENT_RADIUS)
        slider.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
