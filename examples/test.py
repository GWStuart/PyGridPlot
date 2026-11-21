import pygame
import pygridplot as pgp

# import numpy as np

win = pygame.display.set_mode((800, 600))
pygame.display.set_caption("test")

plane = pgp.Plane()

run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run = False

    pygame.display.update()

