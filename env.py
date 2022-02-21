import pygame
from PIL import Image
import random
from actions import *
import numpy as np
import cv2

def rect_intersect(x0, y0, w0, h0, x1, y1, w1, h1):
    if not (x0 <= x1 <= x0 + w0 or x1 <= x0 <= x1 + w1):
        return False
    if not (y0 <= y1 <= y0 + h0 or y1 <= y0 <= y1 + h1):
        return False
    return True

class Policy:
    def get_action(self, agent, env):
        pass

class NaivePolicy:
    def __init__(self):
        self.default_move_count = 20
        self.default_move = MOVE_LEFT

    def get_action(self, agent, env):
        min_dist = float('inf')
        best = None
        for c in env.collectibles:
            if c.is_good:
                dist = (c.x - agent.x) ** 2 + (c.y - agent.y) ** 2
                if agent.can_see(c) and dist < min_dist:
                    min_dist = dist
                    best = c

        if best is None:
            self.default_move_count -= 1
            if self.default_move_count == 0:
                self.default_move = [MOVE_LEFT, MOVE_DOWN, MOVE_UP, MOVE_RIGHT][random.randint(0, 3)]
                self.default_move_count = 20

            return self.default_move

        if best.x < agent.x and not (best.x <= agent.x <= best.x + best.size):
            return MOVE_LEFT
        if best.x > agent.x and not (agent.x <= best.x <= agent.x + agent.size):
            return MOVE_RIGHT
        if best.y < agent.y and not (best.y <= agent.y <= best.y + best.size):
            return MOVE_UP
        if best.y > agent.y and not (agent.y <= best.y <= agent.y + agent.size):
            return MOVE_DOWN

        return NOTHING

class View:
    def __init__(self, player):
        self.factor = 1 / 32 # 64 pixels on screen = 1 world units
        self.player = player
        self.screen_height = int(player.vis_range / self.factor)
        self.screen_width = int(player.vis_range / self.factor)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.player_center = False

    def get_x(self):
        if not self.player_center:
            return 0
        return self.player.x - self.screen_width / 2 * self.factor + self.player.size / 2

    def get_y(self):
        if not self.player_center:
            return 0
        return self.player.y - self.screen_height / 2 * self.factor + self.player.size / 2

    def to_screen_coords(self, x, y):
        _x = self.get_x()
        _y = self.get_y()

        return (x - _x) / self.factor, (y - _y) / self.factor

    def to_screen_size(self, v):
        return v / self.factor

class Drawable:
    def __init__(self, x, y, size):
        self.cached = {}
        self.image = None
        self.size = size
        self.x = x
        self.y = y

    def draw(self, view):
        if view in self.cached:
            scaled = self.cached[view]
        else:
            scaled = pygame.transform.scale(self.image,
                                            (view.to_screen_size(self.size), view.to_screen_size(self.size)))
            self.cached[view] = scaled
        view.screen.blit(scaled, view.to_screen_coords(self.x, self.y))


class Bullet(Drawable):
    def __init__(self, x, y, v_x, v_y, player):
        super().__init__(x, y, size=1)

        self.v_x = v_x
        self.v_y = v_y

        self.image = pygame.image.load('./images/laser.png').convert_alpha()

        self.player = player

    def update(self):
        self.x += self.v_x
        self.y += self.v_y


class Player(Drawable):
    def __init__(self, name, x, y, which):
        super().__init__(x, y, size=1)

        self.orig_x, self.orig_y = x, y
        self.vis_range = 20
        self.view = View(self)

        self.v_x = 0
        self.v_y = 0
        self.name = name

        if which == 1:
            self.image = pygame.image.load('./images/steve.png').convert_alpha()
        else:
            self.image = pygame.image.load('./images/alex.png').convert_alpha()
        self.poison_overlay = pygame.image.load('./images/poison.png').convert_alpha()

        self.score = 0

        self.poisoned = False
        self.poison_timer = 0
        self.POISON_DURATION = 200
        self.delayed_penalties = []

        self.automate = False
        self.env = None
        self.default_policy = NaivePolicy()

    def reset(self):
        self.x = self.orig_x
        self.y = self.orig_y
        self.score = 0
        self.poisoned = False
        self.poison_timer = 0
        self.delayed_penalties = []
        self.v_x = 0
        self.v_y = 0

    def update(self):
        self.poison_timer = max(0, self.poison_timer - 1)
        if self.poison_timer == 0:
            self.poisoned = False

        for i in range(len(self.delayed_penalties)):
            self.delayed_penalties[i] -= 1
        while self.delayed_penalties != [] and self.delayed_penalties[0] == 0:
            self.delayed_penalties.pop(0)
            self.score -= 50

        if self.automate:
            self.env.perform_action(self, self.default_policy.get_action(self, self.env))

    def draw(self, view):
        super().draw(view)
        if self.poisoned:
            scaled = pygame.transform.scale(self.poison_overlay,
                                            (view.to_screen_size(self.size), view.to_screen_size(self.size)))
            view.screen.blit(scaled, view.to_screen_coords(self.x + self.size / 2, self.y - self.size / 2))

    def collect(self, item):
        if isinstance(item, Apple):
            if item.is_good:
                self.score += 10
            else:
                self.poisoned = True
                self.poison_timer = self.POISON_DURATION
                self.delayed_penalties.append(self.POISON_DURATION)

    def punish(self, player):
        if player.poisoned:
            self.score += 200
            player.score -= 200
        else:
            self.score -= 200

    def can_see(self, entity : Drawable):
        c_x, c_y = self.x + self.size / 2, self.y + self.size / 2
        x0, y0 = c_x - self.vis_range / 2, c_y - self.vis_range / 2
        return rect_intersect(x0, y0, self.vis_range, self.vis_range, entity.x, entity.y, entity.size, entity.size)

class Collectible(Drawable):
    def __init__(self, x, y, size):
        super().__init__(x, y, size)

class Apple(Collectible):
    def __init__(self, x, y, is_good):
        super().__init__(x, y, size=1)

        self.is_good = is_good

        if is_good:
            self.image = pygame.image.load('./images/apple_red.png').convert_alpha()
        else:
            self.image = pygame.image.load('./images/apple_blue.png').convert_alpha()

class World:
    def __init__(self, world_width, world_height):
        self.width = world_width
        self.height = world_height

        self.players = []
        self.collectibles = []
        self.projectiles = []

        self.font = pygame.font.SysFont('monospace', 30)

    def register_players(self, players):
        for p in players:
            self.players.append(p)
            p.env = self

    def update(self):
        # spawning
        for i in range(self.height):
            for j in range(self.width):
                if random.randint(0, 1000000) < 45:
                    self.spawn(i, j, random.random() < .8)

        # movement
        for player in self.players:
            player.update()
            player.y = min(max(player.y + player.v_y, 0), self.height - player.size)
            player.x = min(max(player.x + player.v_x, 0), self.width - player.size)

        removals = []
        for projectile in self.projectiles:
            projectile.y = min(max(projectile.y + projectile.v_y, 0), self.height - projectile.size)
            projectile.x = min(max(projectile.x + projectile.v_x, 0), self.width - projectile.size)

            if projectile.y in [0, self.height - projectile.size] or projectile.x in [0, self.width - projectile.size]:
                removals.append(projectile)
        for p in removals:
            self.projectiles.remove(p)

        # collision
        for p in self.players:
            x0, y0, w0, h0 = p.x, p.y, p.size, p.size

            for c in self.collectibles:
                x1, y1, w1, h1 = c.x, c.y, c.size, c.size

                if rect_intersect(x0, y0, w0, h0, x1, y1, w1, h1):
                    p.collect(c)
                    self.collectibles.remove(c)

            for c in self.projectiles:
                if p == c.player:
                    continue

                x1, y1, w1, h1 = c.x, c.y, c.size, c.size

                if rect_intersect(x0, y0, w0, h0, x1, y1, w1, h1):
                    c.player.punish(p)
                    self.projectiles.remove(c)


    def spawn(self, x, y, is_good):
        self.collectibles.append(Apple(x, y, is_good))

    def spawn_projectile(self, x, y, v_x, v_y, player):
        self.projectiles.append(Bullet(x, y, v_x, v_y, player))

    def render(self, view, display_score=False):
        assert(view.player in self.players)

        view.screen.fill((0, 0, 0))
        rect_x0, rect_y0 = view.to_screen_coords(0, 0)
        pygame.draw.rect(view.screen, (116, 89, 128), pygame.Rect(rect_x0, rect_y0, view.to_screen_size(self.width), view.to_screen_size(self.height)))

        for ply in self.players:
            ply.draw(view)

        for collectible in self.collectibles:
            collectible.draw(view)

        for projectile in self.projectiles:
            projectile.draw(view)

        if display_score:
            pygame.display.set_caption('Score : ' + str(view.player.score))

        score_board = []
        for player in self.players:
            score_board.append(f'{player.name} : ' + str(player.score))
        for i, line in enumerate(score_board):
            textsurface = self.font.render(line, True, (255, 255, 255))
            view.screen.blit(textsurface, (30, 30 + i * 30))

        pygame.display.update()

    def perform_action(self, player, action):
        if action == MOVE_UP:
            player.v_y = -.3
            player.v_x = 0
        elif action == MOVE_DOWN:
            player.v_y = .3
            player.v_x = 0
        elif action == MOVE_LEFT:
            player.v_y = 0
            player.v_x = -.3
        elif action == MOVE_RIGHT:
            player.v_y = 0
            player.v_x = .3
        elif action == SHOOT_LEFT:
            self.spawn_projectile(player.x, player.y, -1, 0, player)
        elif action == SHOOT_RIGHT:
            self.spawn_projectile(player.x, player.y, 1, 0, player)
        elif action == SHOOT_UP:
            self.spawn_projectile(player.x, player.y, 0, -1, player)
        elif action == SHOOT_DOWN:
            self.spawn_projectile(player.x, player.y, 0, 1, player)
        elif action == NOTHING:
            player.v_y = 0
            player.v_x = 0

    def screenshot(self, view, out_w, out_h):
        """
        Takes a screenshot of the game , converts it to grayscale, reshapes it to size INPUT_HEIGHT, INPUT_WIDTH,
        and returns a np.array.
        Credits goes to https://github.com/danielegrattarola/deep-q-snake/blob/master/snake.py
        """
        data = pygame.image.tostring(view.screen, 'RGB')  # Take screenshot
        image = Image.frombytes('RGB', (view.screen_width, view.screen_height), data)

        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(image, (out_w, out_h), interpolation = cv2.INTER_AREA)

        # image = image.convert('L')  # Convert to greyscale
        # image = image.resize((out_w, out_h))
        # matrix = np.asarray(image.getdata(), dtype=np.uint8)
        # matrix = (matrix - 128) / (128 - 1)  # Normalize from -1 to 1
        # return matrix.reshape(image.size[0], image.size[1])

    def reset(self):
        self.collectibles = []
        self.projectiles = []

        for player in self.players:
            player.reset()

if __name__ == '__main__':
    import time

    pygame.init()
    pygame.font.init()

    running = True

    env = World(20, 20)

    my_player = Player('Alex', 0, 0, 2)
    my_player.automate = True

    cpu_player = Player('Steve', 19, 19, 1)
    cpu_player.automate = True

    env.register_players([my_player, cpu_player])

    while running:
        time.sleep(0.016)  # Make the game slow down

        env.update()
        env.render(my_player.view, display_score=True)

        keys = pygame.key.get_pressed()  # checking pressed keys
        if keys[pygame.K_UP]:
            env.perform_action(my_player, MOVE_UP)
        elif keys[pygame.K_DOWN]:
            env.perform_action(my_player, MOVE_DOWN)
        elif keys[pygame.K_LEFT]:
            env.perform_action(my_player, MOVE_LEFT)
        elif keys[pygame.K_RIGHT]:
            env.perform_action(my_player, MOVE_RIGHT)
        else:
            env.perform_action(my_player, NOTHING)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    env.perform_action(my_player, SHOOT_LEFT)
                if event.key == pygame.K_d:
                    env.perform_action(my_player, SHOOT_RIGHT)
                if event.key == pygame.K_w:
                    env.perform_action(my_player, SHOOT_UP)
                if event.key == pygame.K_s:
                    env.perform_action(my_player, SHOOT_DOWN)


