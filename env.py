import pygame
from PIL import Image
import random
from actions import *
import numpy as np
import cv2

# https://opengameart.org/content/weird-fruits-16x16

DISCRETE = True

GOOD_APPLE_SCORE = 100
POISON_VALUE = 100
MOVE_SPEED = 1
FROZEN_TIMER = 10
EAT_PENALTY = 0
PLAYER_LIFE = 30000
GROW_TIME = 40
GROW_CHANCE = 30
PUNISH_REWARD = 300
PUNISH_COST = 100
PUNISH_PENALTY = 300
LEFT = 1
RIGHT = 2
POISON_DURATION = 8
PUNISHED_DURATION = 4
P_GOOD = 1 / 3
PADDING = 1
VIEW_FACTOR = 1 / 32

import random as random

def same_position(x0, y0, w0, h0, x1, y1, w1, h1):
    return x0 == x1 and y0 == y1

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
            if isinstance(c, Apple) and c.grow_timer > 0:
                continue
            dist = (c.x - agent.x) ** 2 + (c.y - agent.y) ** 2
            if dist < min_dist:
                min_dist = dist
                best = c

        if best is None:
            return NOTHING

        if best.y < agent.y:
            return MOVE_UP
        if best.y > agent.y:
            return MOVE_DOWN

        return EAT

class View:
    def __init__(self, player, width, height, x_offset, y_offset):
        self.player = player
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.width = width
        self.height = height

        self.range_x = 111#2.5
        self.range_y = 111#2.5

        self.zoom_x = 1
        self.zoom_y = 1

    def viz_rect(self):
        rect_x0, rect_y0 = max(0, self.player.x + .5 - self.range_x), max(0, self.player.y + .5 - self.range_y)
        rect_x1 = min(self.player.env.width, self.player.x + .5 + self.range_x)
        rect_y1 = min(self.player.env.height, self.player.y + .5 + self.range_y)
        w, h = rect_x1 - rect_x0, rect_y1 - rect_y0
        rect_x0, rect_y0 = self.to_screen_coords(rect_x0, rect_y0)
        return rect_x0, rect_y0, w, h

    def can_see(self, x, y):
        w = self.range_x - .5
        h = self.range_y - .5
        return self.player.x - w <= x <= self.player.x + w \
            and self.player.y - h <= y <= self.player.y + h

    def get_x(self):
        return -PADDING / 2

    def get_y(self):
        return -PADDING / 2

    def to_screen_coords(self, x, y):
        _x = self.get_x()
        _y = self.get_y()

        return (x - _x) / VIEW_FACTOR * self.zoom_x + self.x_offset, \
               (y - _y) / VIEW_FACTOR * self.zoom_y + self.y_offset

    def to_screen_size(self, v):
        return v / VIEW_FACTOR

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
        view.player.env.screen.blit(scaled, view.to_screen_coords(self.x, self.y))

def rect_intersect_(x0, y0, w0, h0, x1, y1, w1, h1):
    return (x0 < x1 < x0 + w0 or x1 < x0 < x1 + w1 or x0 == x1) and \
           (y0 < y1 < y0 + h0 or y1 < y0 < y1 + h1 or y0 == y1)

def rect_intersect(obj1 : Drawable, obj2 : Drawable):
    return rect_intersect_(obj1.x, obj1.y, obj1.size, obj1.size, obj2.x, obj2.y, obj2.size, obj2.size)

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
    def __init__(self, name, x, y, which, facing):
        super().__init__(x, y, size=1)

        self.orig_x, self.orig_y = x, y
        self.vis_range_x = 9
        self.vis_range_y = 9
        self.view = None

        self.v_x = 0
        self.v_y = 0
        self.name = name
        self.facing = facing

        self.friendly_image = pygame.image.load('./images/cow.png').convert_alpha()
        self.enemy_image = pygame.image.load('./images/pig.png').convert_alpha()

        # if which == 1:
        #     self.image = pygame.image.load('./images/pig.png').convert_alpha()
        # else:
        #     self.image = pygame.image.load('./images/cow.png').convert_alpha()

        self.poison_overlay = pygame.image.load('./images/poison.png').convert_alpha()
        self.punished_overlay = pygame.image.load('./images/spoon.png').convert_alpha()

        self.score = 0

        self.poisoned = False
        self.poison_timer = 0

        self.delayed_penalties = []

        self.automate = False
        self.env = None
        self.default_policy = NaivePolicy()

        self.frozen_timer = 0

        self.life = PLAYER_LIFE

    def is_frozen(self):
        return self.frozen_timer > 0

    def freeze(self):
        self.frozen_timer = FROZEN_TIMER

    def reset(self):
        self.x = self.orig_x
        self.y = self.orig_y
        self.score = 0
        self.poisoned = False
        self.poison_timer = 0
        self.punished = False
        self.punished_timer = 0
        self.delayed_penalties = []
        self.v_x = 0
        self.v_y = 0
        self.life = PLAYER_LIFE

    def update(self):
        if not DISCRETE:
            self.x = min(max(self.x + self.v_x, 0), self.env.width - self.size)

            for c in self.env.collectibles:
                if rect_intersect(self, c):
                    self.x = c.x - self.size if self.v_x > 0 else c.x + c.size
                    break

            self.y = min(max(self.y + self.v_y, 0), self.env.height - self.size)

            for c in self.env.collectibles:
                if rect_intersect(self, c):
                    self.y = c.y - self.size if self.v_y > 0 else c.y + c.size
                    break


        self.poison_timer = max(0, self.poison_timer - 1)
        self.punished_timer = max(0, self.punished_timer - 1)
        self.frozen_timer = max(0, self.frozen_timer - 1)

        if self.poison_timer == 0:
            self.poisoned = False
        if self.punished_timer == 0:
            self.punished = False

        for i in range(len(self.delayed_penalties)):
            self.delayed_penalties[i] -= 1
        while self.delayed_penalties != [] and self.delayed_penalties[0] == 0:
            self.delayed_penalties.pop(0)
            self.score -= POISON_VALUE

    def perform_automatic(self):
        if self.automate:
            self.env.perform_action(self, self.default_policy.get_action(self, self.env))

    def draw(self, view):
        self.image = self.friendly_image if view.player == self else self.enemy_image

        super().draw(view)

        if self.poisoned:
            scaled = pygame.transform.scale(self.poison_overlay,
                                            (view.to_screen_size(self.size), view.to_screen_size(self.size)))
            view.player.env.screen.blit(scaled, view.to_screen_coords(self.x + self.size / 2, self.y - self.size / 2))

        if self.punished:
            scaled = pygame.transform.scale(self.punished_overlay,
                                            (view.to_screen_size(self.size), view.to_screen_size(self.size)))
            view.player.env.screen.blit(scaled, view.to_screen_coords(self.x - self.size / 2, self.y - self.size / 2))


    def collect(self, item):
        if isinstance(item, Apple):
            if item.grow_timer > 0:
                return
            if item.is_good:
                self.score += GOOD_APPLE_SCORE
                self.life = PLAYER_LIFE
            else:
                self.poisoned = True
                self.poison_timer = POISON_DURATION
                self.delayed_penalties.append(POISON_DURATION)
            item.on_collect()

    def on_punished(self):
        self.punished = True
        self.punished_timer = PUNISHED_DURATION

    def punish(self, player):
        if not self.view.can_see(player.x, player.y):
            return

        if player.poisoned and not player.punished:
            self.score += PUNISH_REWARD - PUNISH_COST
            player.score -= PUNISH_PENALTY
            player.on_punished()
        else:
            self.score -= PUNISH_COST

    def can_see(self, entity : Drawable):
        # TODO: broken

        c_x, c_y = self.x + self.size / 2, self.y + self.size / 2
        x0, y0 = c_x - self.vis_range_x, c_y - self.vis_range_y
        return rect_intersect(x0, y0, self.vis_range_x * 2, self.vis_range_y * 2, entity.x, entity.y, entity.size, entity.size)

class Collectible(Drawable):
    def __init__(self, x, y, size):
        super().__init__(x, y, size)

    def update(self):
        pass

    def on_collect(self):
        pass

class Apple(Collectible):
    def __init__(self, x, y, is_good):
        super().__init__(x, y, size=1)

        self.is_good = is_good

        if is_good:
            self.image = pygame.image.load('./images/fruit1.png').convert_alpha()
        else:
            self.image = pygame.image.load('./images/fruit2.png').convert_alpha()

        self.grow_timer = 0

    def update(self):
        if self.grow_timer > 0:
            if random.randint(0, GROW_CHANCE) == 1:
                self.grow_timer = 0

    def draw(self, view):
        rect_x0, rect_y0 = view.to_screen_coords(self.x, self.y)
        pygame.draw.rect(view.player.env.screen, (205, 170, 140), pygame.Rect(rect_x0, rect_y0, view.to_screen_size(self.size), view.to_screen_size(self.size)))

        if self.grow_timer > 0:
            return
        super().draw(view)

    def on_collect(self):
        self.grow_timer = GROW_TIME

class World:
    def __init__(self, world_width, world_height, n_players, start_positions):
        self.width = world_width
        self.height = world_height

        self.n_players = n_players
        self.start_positions = start_positions

        self.width_pixel_view = int((PADDING + self.width) / VIEW_FACTOR)
        self.height_pixel_view = int((PADDING + self.height) / VIEW_FACTOR)
        self.screen_height = self.height_pixel_view
        self.screen_width = int((PADDING + self.width) / VIEW_FACTOR) * n_players
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        self.players = []
        self.collectibles = []
        self.projectiles = []

        self.font = pygame.font.SysFont('monospace', 30)

        self.spawn_timer = 0

        self.temp = True
        self.kill = False
        self.b = True

        self.bg_tile = pygame.image.load('./images/grass.png').convert_alpha()
        self.rgb_value = 0

        self.ordering = None

        self.init_collectibles()

    def init_collectibles(self):
        #random.seed(123)

        type = random.randint(0, 2)
        if type == 0:
            arr = [3, 3, 3]
        elif type == 1:
            arr = [8, 3, 1]
        else:
            arr = [1, 3, 8]

        self.collectibles = []
        i = 0
        for j, x in enumerate([0, 4, 8]):
            i += 1
            idx_start = 1 if j == 1 else 0
            idx_end = self.height - 2 if j == 1 else self.height - 1
            #bads = np.random.choice(np.arange(idx_start, idx_end + 1), arr[j], replace=False)
            for i in range(idx_start, idx_end + 1):
                y = i
                apple = Apple(x, y, random.random() < P_GOOD)#True if i not in bads else False)
                apple.grow_timer = 0 if random.randint(0, 10) == 1 else 1
                self.collectibles.append(apple)

    def register_players(self, players):
        assert(len(players)) == self.n_players

        for i, p in enumerate(players):
            self.players.append(p)
            p.env = self
            p.view = View(p, self.width_pixel_view, self.height_pixel_view, x_offset=i * self.width_pixel_view, y_offset=0)

        self.set_player_positions()

    def set_player_positions(self):
        np.random.shuffle(self.start_positions)
        for i, p in enumerate(self.players):
            p.orig_x, p.orig_y = self.start_positions[i]

        # pos = []
        # for p in self.players:
        #     while True:
        #         flag = False
        #         p.orig_x = random.randint(0, self.width - 1)
        #         p.orig_y = random.randint(0, self.height - 1)
        #         for c in self.collectibles:
        #             if c.x == p.orig_x and c.y == p.orig_y:
        #                 flag = True
        #         if flag or (p.orig_x, p.orig_y) in pos:
        #             continue
        #         pos.append((p.orig_x, p.orig_y))
        #         break

    def update(self):
        self.rgb_value = (self.rgb_value + 256 * 256 + 2 * 256 + 3 * 1) % (256 ** 3)

        for c in self.collectibles:
            c.update()

        # movement
        for player in self.players:
            player.update()

    def spawn(self, x, y, is_good):
        self.collectibles.append(Apple(x, y, is_good))

    def spawn_projectile(self, x, y, v_x, v_y, player):
        self.projectiles.append(Bullet(x, y, v_x, v_y, player))

    def get_rgb(self):
        r = self.rgb_value // (256 ** 2)
        g = (self.rgb_value // 256) % 256
        b = self.rgb_value % 256
        print (r,g,b)
        return (r, g, b)

    def render(self, view, display_score=False):
        assert(view.player in self.players)



        rect_x0, rect_y0 = view.to_screen_coords(-PADDING/2, -PADDING/2)
        pygame.draw.rect(self.screen, (255, 204, 153), pygame.Rect(rect_x0, rect_y0, view.to_screen_size(self.width + PADDING), view.to_screen_size(self.height + PADDING)))

        rect_x0, rect_y0 = view.to_screen_coords(0, 0)
        color = (0, 0, 0)
        pygame.draw.rect(self.screen, color, pygame.Rect(rect_x0, rect_y0,
                                                                   view.to_screen_size(self.width),
                                                                   view.to_screen_size(self.height)))

        rect_x0, rect_y0, rect_w, rect_h = view.viz_rect()
        color = (204, 255, 255)
        pygame.draw.rect(self.screen, color, pygame.Rect(rect_x0, rect_y0,
                                                                   view.to_screen_size(rect_w),
                                                                   view.to_screen_size(rect_h)))

        for collectible in self.collectibles:
            if view.can_see(collectible.x, collectible.y):
                collectible.draw(view)

        for ply in self.players:
            if view.can_see(ply.x, ply.y):
                ply.draw(view)

        pygame.display.set_caption('Score : ' + str(view.player.score) + 'Life: ' + str(view.player.life))

        if display_score:
            score_board = []
            for player in self.players:
                score_board.append(f'{player.name} : ' + str(player.score))
            for i, line in enumerate(score_board):
                textsurface = self.font.render(line, True, (255, 255, 255))
                self.screen.blit(textsurface, (30, 30 + i * 30))

        pygame.display.update()

    def perform_action(self, player, action):
        # if action == MOVE_UP and player.y == 0:
        #     player.y = 0
        #     player.x = (player.x - 3) % 9
        #     return
        # elif action == MOVE_DOWN and player.y == self.height - 1:
        #     player.y = self.height - 1
        #     player.x = (player.x + 3) % 9
        #     return

        if action == MOVE_UP:
            player.v_y = -MOVE_SPEED
            player.v_x = 0
        elif action == MOVE_DOWN:
            player.v_y = MOVE_SPEED
            player.v_x = 0
        elif action == MOVE_LEFT:
            player.v_y = 0
            player.v_x = -MOVE_SPEED
        elif action == MOVE_RIGHT:
            player.v_y = 0
            player.v_x = MOVE_SPEED
        else:
            player.v_y = 0
            player.v_x = 0

        if DISCRETE:
            x_ = player.x + player.v_x
            y_ = player.y + player.v_y

            for c in self.collectibles:
                if c.x == x_ and c.y == y_:
                    return

        if action == EAT:
            if DISCRETE:
                for c in self.collectibles:
                    if c.x == player.x and abs(c.y - player.y) == 1:
                        player.collect(c)
                    elif c.y == player.y and abs(c.x - player.x) == 1:
                        player.collect(c)
            else:
                min_dist = float('inf')
                closest = None
                for c in self.collectibles:
                    if not isinstance(c, Apple):
                        continue
                    dist = (player.x - c.x) ** 2 + (player.y - c.y) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        closest = c
                if closest is not None and ((closest.x - player.x) ** 2 + (closest.y - player.y) ** 2) ** 0.5 < 1.1:
                    player.collect(closest)

        if action == PUNISH:
            other_player = [p for p in self.players if p != player][0]
            player.punish(other_player)

        if DISCRETE:
            player.y = min(max(player.y + player.v_y, 0), self.height - player.size)
            player.x = min(max(player.x + player.v_x, 0), self.width - player.size)


    def get_score(self, player : Player):
        return player.score

    def screenshot(self, view, out_w, out_h):
        data = pygame.image.tostring(self.screen, 'RGB')
        image = Image.frombytes('RGB', (self.screen_width, self.screen_height), data)

        image = np.asarray(image)[:, view.x_offset : view.x_offset + view.width]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(image, (out_w, out_h), interpolation = cv2.INTER_AREA)

    def screenshot_basic(self, view):
        M = np.zeros((self.width, self.height))

        for c in self.collectibles:
            M[c.y, c.x] = 2 if c.is_good else 3

        for player in self.players:
            M[player.y, player.x] = 1

        return M

    def screenshot_positions(self):
        good = [x for x in self.collectibles if x.is_good][0]

        arr = [
            self.players[0].x,
            self.players[0].y,
            good.x,
            good.y,
            self.players[0].life
        ]

        return np.array([arr])

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.init_collectibles()
        self.projectiles = []

        for player in self.players:
            player.reset()

        self.kill = False
        self.b = True
        self.position_pointer = 0

        self.ordering = np.arange(len(self.players))
        np.random.shuffle(self.ordering)

        self.set_player_positions()

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


