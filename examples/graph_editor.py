"""A small expression list: type maths, drag points, save and load.

Type a line of maths in the box at the bottom left and press ``Enter`` to
add it to the graph:

* ``y = x^2`` or ``f(x) = sin(x) / x``: plots (a named one is also a
  function the other lines can use, as ``f(2)`` or ``f'(x)``);
* ``r = 1 + cos(θ)``: a polar curve; ``(cos(t), sin(2t))``: a parametric
  curve;
* ``x^2 + y^2 = 9``: an implicit curve; ``a = 2``: a parameter;
* ``P = (1, 2)``: a point you can drag (with numbers), or ``Q = (a, f(a))``:
  a point that follows what it uses;
* ``g(u, v) = u v``: a function for the other lines.

Lines can use each other's names in any order. A line with a mistake is
listed with the reason in red and is not drawn; a line that cannot be read
at all is refused, with the reason under the box.

Keys (``F1`` shows them): ``Enter`` adds the line, ``Up``/``Down`` select a
line, ``Ctrl+E`` edits the selected line (``Enter`` then replaces it),
``Delete`` removes it, ``Ctrl+H`` hides or shows it, ``Escape`` clears the
box. ``Ctrl+S`` saves the graph (and its view) to ``graph.json`` in the
current folder, ``Ctrl+L`` loads it back, ``Ctrl+N`` starts again. Pan and
zoom with the mouse; the keyboard is for typing.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pygame

import pygridplot as pgp
from pygridplot.graph import (
    DefinitionItem,
    DraggableItem,
    ImplicitItem,
    Item,
    ParameterItem,
    ParametricItem,
    PlotItem,
    PointItem,
    PolarItem,
    StyledItem,
)
from pygridplot.render.fonts import bundled_font_path

SIDEBAR = 380
"""Width of the expression list, in pixels."""

SAVE_PATH = Path("graph.json")
"""Where Ctrl+S saves and Ctrl+L loads."""

INK, FAINT, ERROR, PANEL = (35, 35, 40), (150, 150, 155), (190, 40, 40), (246, 246, 248)

HELP = [
    "Enter: add the line (or replace the edited one)",
    "Up / Down: select a line",
    "Ctrl+E: edit it    Delete: remove it    Ctrl+H: hide or show it",
    "Escape: clear the box",
    f"Ctrl+S: save to {SAVE_PATH}    Ctrl+L: load it    Ctrl+N: start again",
    "Mouse: drag points, pan and zoom",
]


def starter() -> pgp.Graph:
    """Return the graph the editor opens with."""
    graph = pgp.Graph()
    graph.add("S = (-6, 1.5)", draggable=True, lock="x", y_range=(0.25, 3))
    graph.add("a = S.y")
    graph.add("f(x) = a sin(x)")
    graph.draggable(curve="f", t=2.0, name="P")
    graph.add("y = f'(P.x)(x - P.x) + P.y", width=1.5)
    graph.add("x^2 + y^2 = 9", opacity=0.6)
    graph.add("r = 1 + cos(θ)", visible=False)
    return graph


def add_line(graph: pgp.Graph, text: str) -> Item:
    """Add a typed line; a point with numbers becomes a draggable point."""
    try:
        return graph.add(text, draggable=True)
    except pgp.GraphError:  # not a point of numbers: add it as it is
        return graph.add(text)


def show(value: Any) -> str:
    """Write an expression (text, a number or a Python function) as text."""
    if isinstance(value, str):
        return value
    if callable(value):
        return getattr(value, "__name__", "<function>")
    return f"{value:g}"


def describe(item: Item) -> str:
    """Write an item as the line of maths that makes it."""
    name = item.name
    if isinstance(item, PlotItem):
        lhs = f"{name}(x)" if name else "y"
        return f"{lhs} = {show(item.expression)}"
    if isinstance(item, ParametricItem):
        pair = f"({show(item.x)}, {show(item.y)})"
        return f"{name} = {pair}" if name else pair
    if isinstance(item, PolarItem):
        return f"r = {show(item.r)}"
    if isinstance(item, ImplicitItem):
        level = item.level
        levels = (
            ", ".join(f"{v:g}" for v in level)
            if isinstance(level, tuple)
            else f"{level:g}"
        )
        return f"{show(item.expression)} = {levels}"
    if isinstance(item, PointItem):
        x, y = item.position
        pair = f"({show(x)}, {show(y)})"
        return f"{name} = {pair}" if name else pair
    if isinstance(item, DraggableItem):
        where = f"({item.x:.3g}, {item.y:.3g})" if item.position else "(not drawn yet)"
        label = f"{name} = " if name else ""
        if isinstance(item.curve, str):
            return f"{label}{where} on {item.curve}"
        return f"{label}{where}"
    if isinstance(item, ParameterItem):
        return f"{item.name} = {show(item.value)}"
    if isinstance(item, DefinitionItem):
        return f"{item.name}({', '.join(item.parameters)}) = {item.expression}"
    return f"{item.kind} ({name or 'unnamed'})"


def fit(font: pygame.font.Font, text: str, width: int) -> str:
    """Shorten text with an ellipsis so that it fits a width."""
    if font.size(text)[0] <= width:
        return text
    while text and font.size(text + "…")[0] > width:
        text = text[:-1]
    return text + "…"


class Editor:
    """The expression list: the graph, the typed line, the selection."""

    def __init__(self, plane: pgp.Plane) -> None:
        self.plane = plane
        self.graph = starter()
        self.text = ""
        self.editing: Item | None = None
        self.selected: int | None = None
        self.status = ("", INK)
        self.help = False
        self.rows: list[tuple[pygame.Rect, int]] = []

    # --- Commands ----------------------------------------------------------
    def submit(self) -> None:
        """Add the typed line, or replace the line being edited with it."""
        text = self.text.strip()
        if not text:
            return
        graph, old = self.graph, self.editing
        index = graph.items.index(old) if old is not None else None
        if old is not None:
            graph.remove(old)  # frees its name for the new line
        try:
            item = add_line(graph, text)
        except (pgp.ExpressionError, pgp.GraphError, ValueError, TypeError) as error:
            if old is not None:
                graph.append(old, index=index)
            message = (
                error.message if isinstance(error, pgp.ExpressionError) else str(error)
            )
            self.status = (message, ERROR)
            return
        if index is not None:
            graph.move(item, index)
        self.text, self.editing = "", None
        self.selected = graph.items.index(item)
        self.status = ("", INK)

    def selection(self) -> Item | None:
        """Return the selected item, if any."""
        items = self.graph.items
        if self.selected is None or not items:
            return None
        self.selected = min(self.selected, len(items) - 1)
        return items[self.selected]

    def new_graph(self, graph: pgp.Graph) -> None:
        """Replace the graph (taking the old one's points off the plane)."""
        self.graph.detach(self.plane)
        self.graph = graph
        self.selected = self.editing = None
        graph.apply_view(self.plane)

    def key(self, event: pygame.event.Event) -> None:
        """Handle a key press."""
        ctrl = bool(event.mod & pygame.KMOD_CTRL)
        item = self.selection()
        count = len(self.graph.items)
        if event.key == pygame.K_F1:
            self.help = not self.help
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.submit()
        elif event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        elif event.key == pygame.K_ESCAPE:
            self.text, self.editing, self.status = "", None, ("", INK)
        elif event.key == pygame.K_UP and count:
            self.selected = (
                count - 1 if self.selected is None else max(0, self.selected - 1)
            )
        elif event.key == pygame.K_DOWN and count:
            self.selected = (
                0 if self.selected is None else min(count - 1, self.selected + 1)
            )
        elif event.key == pygame.K_DELETE and item is not None:
            self.graph.remove(item)
            if item is self.editing:
                self.editing = None
        elif ctrl and event.key == pygame.K_e and item is not None:
            self.text, self.editing = describe(item), item
        elif ctrl and event.key == pygame.K_h and item is not None:
            item.visible = not item.visible
        elif ctrl and event.key == pygame.K_s:
            self.graph.capture_view(self.plane)
            try:
                self.graph.save(SAVE_PATH)
                self.status = (f"saved to {SAVE_PATH}", INK)
            except (pgp.GraphError, OSError) as error:
                self.status = (str(error), ERROR)
        elif ctrl and event.key == pygame.K_l:
            try:
                self.new_graph(pgp.Graph.load(SAVE_PATH))
                self.status = (f"loaded {SAVE_PATH}", INK)
            except (pgp.GraphError, OSError) as error:
                self.status = (str(error), ERROR)
        elif ctrl and event.key == pygame.K_n:
            self.new_graph(pgp.Graph())

    def click(self, pos: tuple[int, int]) -> None:
        """Select the line under a click in the list."""
        for rect, index in self.rows:
            if rect.collidepoint(pos):
                self.selected = index

    # --- Drawing ---------------------------------------------------------
    def draw_list(self, screen: pygame.Surface, fonts: tuple[Any, Any]) -> None:
        """Draw the expression list, the input box and the status line."""
        font, small = fonts
        height = screen.get_height()
        pygame.draw.rect(screen, PANEL, (0, 0, SIDEBAR, height))
        pygame.draw.line(
            screen, (210, 210, 215), (SIDEBAR - 1, 0), (SIDEBAR - 1, height)
        )
        screen.blit(font.render("Graph   (F1: keys)", True, FAINT), (14, 12))
        palette = self.plane.theme.palette
        errors = self.graph.errors
        y, self.rows = 44, []
        for index, item in enumerate(self.graph.items):
            if y > height - 130:
                screen.blit(small.render("…", True, FAINT), (14, y))
                break
            message = errors.get(item)
            row = pygame.Rect(6, y - 3, SIDEBAR - 12, 24 + (18 if message else 0))
            self.rows.append((row, index))
            if index == self.selected:
                pygame.draw.rect(screen, (225, 232, 245), row, border_radius=4)
            if isinstance(item, StyledItem):
                color = (
                    item.color
                    if item.color is not None
                    else palette[(item.palette or 0) % len(palette)]
                )
                swatch = pygame.Rect(14, y + 4, 10, 10)
                if item.visible:
                    pygame.draw.rect(
                        screen, pygame.Color(color), swatch, border_radius=2
                    )
                else:
                    pygame.draw.rect(screen, FAINT, swatch, 1, border_radius=2)
            ink = INK if item.visible else FAINT
            text = fit(font, describe(item), SIDEBAR - 52)
            screen.blit(font.render(text, True, ink), (32, y))
            if message:
                note = fit(small, message, SIDEBAR - 52)
                screen.blit(small.render(note, True, ERROR), (32, y + 20))
            y += row.height + 4
        box = pygame.Rect(10, height - 76, SIDEBAR - 20, 32)
        pygame.draw.rect(screen, (255, 255, 255), box, border_radius=4)
        pygame.draw.rect(screen, (120, 140, 190), box, 1, border_radius=4)
        prompt = "edit> " if self.editing is not None else "> "
        caret = "|" if pygame.time.get_ticks() // 500 % 2 == 0 else " "
        shown = fit(font, prompt + self.text, box.width - 24)
        screen.blit(font.render(shown + caret, True, INK), (box.x + 8, box.y + 7))
        message, color = self.status
        if message:
            screen.blit(
                small.render(fit(small, message, SIDEBAR - 24), True, color),
                (12, height - 36),
            )

    def draw_help(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        """Draw the list of keys over the plane."""
        texts = [font.render(line, True, INK) for line in HELP]
        width = max(t.get_width() for t in texts) + 24
        panel = pygame.Surface((width, 20 * len(texts) + 16), pygame.SRCALPHA)
        panel.fill((255, 255, 255, 235))
        for number, text in enumerate(texts):
            panel.blit(text, (12, 8 + 20 * number))
        screen.blit(panel, (SIDEBAR + 16, 16))


def main() -> None:
    """Run the example."""
    pygame.init()
    screen = pygame.display.set_mode((1200, 760), pygame.RESIZABLE)
    pygame.display.set_caption("PyGridPlot: graph editor")
    pygame.key.start_text_input()
    fonts = (
        pygame.font.Font(bundled_font_path(), 15),
        pygame.font.Font(bundled_font_path(), 12),
    )
    plane = pgp.Plane(pygame.Rect(SIDEBAR, 0, 1200 - SIDEBAR, 760), x_range=(-8, 8))
    editor = Editor(plane)
    warnings.simplefilter("ignore", pgp.GraphWarning)  # errors are listed instead
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.TEXTINPUT:
                editor.text += event.text
            elif event.type == pygame.KEYDOWN:
                editor.key(event)  # the keyboard types: it does not pan the plane
                continue
            elif event.type == pygame.MOUSEBUTTONDOWN and event.pos[0] < SIDEBAR:
                editor.click(event.pos)
            plane.handle_event(event)

        width, height = screen.get_size()
        wanted = pygame.Rect(SIDEBAR, 0, max(1, width - SIDEBAR), height)
        if (plane.rect.x, plane.rect.y, plane.rect.width, plane.rect.height) != tuple(
            wanted
        ):
            plane.rect = wanted
        plane.draw_grid(screen)
        editor.graph.draw(plane, screen)
        editor.draw_list(screen, fonts)
        if editor.help:
            editor.draw_help(screen, fonts[0])
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
