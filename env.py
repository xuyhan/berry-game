import pygame
from PIL import Image
import random

NOTHING = 0
MOVE_UP = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
MOVE_RIGHT = 4
SHOOT_UP = 5
SHOOT_DOWN = 6
SHOOT_LEFT = 7
SHOOT_RIGHT = 8

def rect_intersect(x0, y0, w0, h0, x1, y1, w1, h1):
    if not (x0 <= x1 <= x0 + w0 or x1 <= x0 <= x1 + w1):
        return False
    if not (y0 <= y1 <= y0 + h0 or y1 <= y0 <= y1 + h1):
        return False
    return True

class View:
    def __init__(self, screen, screen_width, screen_height, player):
        self.screen = screen
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.player = player

        self.factor = 1 / 64 # 64 pixels on screen = 1 world units

    def get_x(self):
        return self.player.x - self.screen_width / 2 * self.factor + self.player.size / 2

    def get_y(self):
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
    def __init__(self, x, y):
        super().__init__(x, y, size=1)

        self.vis_range = 100
        self.v_x = 0
        self.v_y = 0

        self.image = pygame.image.load('./images/steve.png').convert_alpha()
        self.poison_overlay = pygame.image.load('./images/poison.png').convert_alpha()

        self.score = 0

        self.poisoned = False
        self.poison_timer = 0
        self.POISON_DURATION = 200
        self.delayed_penalties = []

    def update(self):
        self.poison_timer = max(0, self.poison_timer - 1)
        if self.poison_timer == 0:
            self.poisoned = False

        for i in range(len(self.delayed_penalties)):
            self.delayed_penalties[i] -= 1
        while self.delayed_penalties != [] and self.delayed_penalties[0] == 0:
            self.delayed_penalties.pop(0)
            self.score -= 50

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
        else:
            self.score -= 200

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

    def update(self):
        # spawning
        for i in range(self.height):
            for j in range(self.width):
                if random.randint(0, 1000000) < 15:
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

if __name__ == '__main__':
    import time

    pygame.init()

    screen = pygame.display.set_mode((800, 800))

    running = True

    env = World(32, 32)
    my_player = Player(11, 11)
    env.players = [my_player, Player(15, 15)]

    view = View(screen, 800, 800, my_player)


    while running:
        time.sleep(0.016)  # Make the game slow down

        env.update()
        env.render(view, display_score=True)

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


