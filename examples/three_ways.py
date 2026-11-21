"""The three ways to use PyGridPlot, side by side.

The window shows three planes, and each draws the *same* small picture: a
sine wave whose crest you set by dragging a point. Only the way the
picture is described differs:

1. **Immediate mode** (left): call the plane's drawing methods every frame.
2. **Objects** (middle): make each part once, change it through its
   attributes, and let the plane draw it.
3. **A graph** (right): write the parts as lines of maths, which can use
   each other's names and be saved to a file.

Drag the point on any plane to reshape its wave (the three are separate
pictures, so they move independently). Press ``Space`` to change the
wave's colour in all three, and see how each way does it. ``S`` prints the
graph as JSON, something only the third way can do. Pan and zoom as usual.

A quick comparison (more in the comments below):

==================  ===================  ===================  ===================
                    Immediate            Objects              Graph
==================  ===================  ===================  ===================
Each frame          draw everything      ``plane.draw()``     ``plane.draw()``
Change something    draw it differently  set an attribute     set an attribute
Parts refer to      Python variables     Python variables     names in the text
each other by
Written in          Python               Python               maths as text
Save to a file      no                   no                   ``graph.save()``
Best for            pictures that        pictures made of     maths typed or
                    change every frame   lasting parts        loaded at run time
==================  ===================  ===================  ===================

All three draw through the same plotting code (the objects and the graph
call the immediate methods for you), so they look identical, cost the
same, and mix freely: one plane can use all three at once.
"""

from __future__ import annotations

import numpy as np
import pygame

import pygridplot as pgp
from pygridplot.render.fonts import bundled_font_path

PANEL = (380, 420)
"""The size of each plane, in pixels."""

GAP, TOP = 16, 40
"""Space between the planes, and above them for their titles."""

COLOURS = ("crimson", "royalblue")
"""The wave's colours, swapped with Space."""

POINT_COLOUR = "darkorange"


def panel_rect(index: int) -> pygame.Rect:
    """Return the screen rectangle of the plane in column ``index``."""
    return pygame.Rect(GAP + index * (PANEL[0] + GAP), TOP, *PANEL)


# ----------------------------------------------------------------------
# 1. Immediate mode
# ----------------------------------------------------------------------
# Nothing is kept except the plane: every frame, the program draws what it
# wants to see, and the plane recomputes it from the current view. To
# change the picture, draw something different next frame; there is
# nothing to update or delete.
#
# The one thing that must persist between frames is a draggable point,
# because the user moves it: `plane.draggable` makes one, and the program
# reads its position each frame.


class ImmediateMode:
    """The picture drawn with the plane's methods, every frame."""

    def __init__(self, plane: pgp.Plane) -> None:
        self.plane = plane
        # A draggable point is made once. The plane drags it when you call
        # plane.handle_event, and we read .x and .y every frame.
        self.crest = plane.draggable((1.5, 2.0), color=POINT_COLOUR)
        # The wave's colour is just a variable that the drawing call reads.
        self.colour = COLOURS[0]

    def swap_colour(self) -> None:
        """Change the wave's colour: change the variable the next call uses."""
        self.colour = COLOURS[1] if self.colour == COLOURS[0] else COLOURS[0]

    def draw(self, screen: pygame.Surface) -> None:
        """Draw one frame. Everything is drawn again, in stacking order."""
        plane, crest = self.plane, self.crest
        plane.draw_grid(screen)  # first: the background, grid and labels
        # The wave is a plain function, made anew each frame. It reads the
        # point's position, so it follows the drag automatically.
        plane.plot(
            screen,
            lambda x: crest.y * np.sin(x - crest.x + np.pi / 2),
            color=self.colour,
            width=3,
        )
        # Draggable points are drawn last, so that they sit on top.
        plane.draw_draggables(screen)


# ----------------------------------------------------------------------
# 2. Objects
# ----------------------------------------------------------------------
# Each part is made once, as an object holding the same arguments the
# drawing method takes (pgp.Function for plane.plot, pgp.Point for points,
# and so on). The plane keeps the objects (plane.add) and draws them all
# with one call per frame (plane.draw). To change the picture, assign to an
# attribute: f.color = "red", f.func = np.cos, p.position = (1, 2), ...
#
# Compared with immediate mode:
# - the frame loop shrinks to plane.draw(screen);
# - each part has an identity you can keep, change, hide (visible = False)
#   or remove, which suits pictures made of lasting parts;
# - objects without a colour keep a stable palette colour (in immediate
#   mode colours are handed out in drawing order, frame by frame);
# - parts still refer to each other through ordinary Python: the wave's
#   function reads the point's attributes. The plane updates every object
#   before drawing any, so the wave sees this frame's drag even though it
#   is drawn before the point.


class Objects:
    """The picture made of objects, once."""

    def __init__(self, plane: pgp.Plane) -> None:
        # The same draggable point, as an object.
        self.crest = pgp.Point((1.5, 2.0), draggable=True, color=POINT_COLOUR)
        crest = self.crest
        # The wave: the function is the same lambda as in immediate mode.
        # A Function is also callable itself: self.wave(1.0) evaluates it.
        self.wave = pgp.Function(
            lambda x: crest.y * np.sin(x - crest.x + np.pi / 2),
            color=COLOURS[0],
            width=3,
        )
        # The plane keeps them and draws them in this order (the draggable
        # point's handle is drawn on top whatever the order).
        plane.add(self.wave, self.crest)

    def swap_colour(self) -> None:
        """Change the wave's colour: assign to its attribute, once."""
        self.wave.color = COLOURS[1] if self.wave.color == COLOURS[0] else COLOURS[0]


# ----------------------------------------------------------------------
# 3. A graph
# ----------------------------------------------------------------------
# A graph is a list of the same objects, written as lines of maths, as in
# a graphing calculator. graph.add reads a line and returns the object it
# makes ("y = ..." gives a pgp.Function, "P = (1, 2)" a pgp.Point), so
# everything about objects still applies: attributes, plane.add, ...
#
# What a graph adds:
# - text: "y = P.y sin(x - P.x + pi/2)" instead of a Python lambda. Text is
#   compiled to NumPy (never run as Python), so it can come from a user or
#   a file;
# - names: objects refer to each other by name (P), in any order; a
#   mistake (an unknown name, a circular definition) is reported on the
#   object that has it, and the rest of the graph still draws;
# - saving: graph.save("wave.json") and pgp.Graph.load(...), including
#   where the user dragged the points.
#
# The price: expressions are the expression language, not Python (Python
# functions can still be registered with graph.register, but must be
# registered again when a saved graph is loaded).


class AGraph:
    """The picture written as maths."""

    def __init__(self, plane: pgp.Plane) -> None:
        self.graph = pgp.Graph()
        # A point with numbers can be made draggable from a line.
        self.graph.add("P = (1.5, 2)", draggable=True, color=POINT_COLOUR)
        # The wave uses P by name: P.x and P.y are its coordinates. Lines
        # could be added in any order; a graph works the order out itself.
        self.wave = self.graph.add(
            "y = P.y sin(x - P.x + pi/2)", color=COLOURS[0], width=3
        )
        # A whole graph can be added to a plane like a single object.
        plane.add(self.graph)

    def swap_colour(self) -> None:
        """Change the wave's colour: the wave is an object, so as for objects."""
        wave = self.graph.items[1]  # or keep the object add returned, as here
        assert wave is self.wave
        assert isinstance(wave, pgp.Function)
        wave.color = COLOURS[1] if wave.color == COLOURS[0] else COLOURS[0]

    def print_json(self) -> None:
        """Print the graph as it would be saved (graph.save writes the same)."""
        print(self.graph.to_json())


# ----------------------------------------------------------------------
# The program: one window, three planes
# ----------------------------------------------------------------------


def main() -> None:
    """Run the example."""
    pygame.init()
    size = (3 * PANEL[0] + 4 * GAP, TOP + PANEL[1] + GAP)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("PyGridPlot: three ways")
    font = pygame.font.Font(bundled_font_path(), 15)

    # Three planes, each in its own rectangle of the window. A plane only
    # reacts to the mouse inside its rectangle, so dragging on one leaves
    # the others alone.
    planes = [pgp.Plane(panel_rect(i), x_range=(-5, 5)) for i in range(3)]
    immediate = ImmediateMode(planes[0])
    objects = Objects(planes[1])
    graph = AGraph(planes[2])
    titles = ["1. Immediate mode", "2. Objects", "3. A graph"]

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # The same change, made the way each style makes changes.
                immediate.swap_colour()
                objects.swap_colour()
                graph.swap_colour()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                graph.print_json()
            # Every plane sees every event (it ignores mouse events outside
            # its rectangle): pan, zoom and drag work in all three ways.
            for plane in planes:
                plane.handle_event(event)

        screen.fill((235, 235, 238))
        for index, title in enumerate(titles):
            rect = panel_rect(index)
            screen.blit(font.render(title, True, (40, 40, 40)), (rect.x, 12))

        # Drawing a frame:
        immediate.draw(screen)  # 1. draw everything, every frame, by hand
        planes[1].draw(screen)  # 2. the plane draws its objects...
        planes[2].draw(screen)  # 3. ...and a graph is drawn the same way

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
