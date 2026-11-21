# Interaction

A plane has pan and zoom controls built in. They work as soon as you pass
events to `plane.handle_event`, and they can be configured, limited or
switched off.

## The controls

| Input | Action |
|---|---|
| Left-drag | Pan. The point you grabbed stays exactly under the cursor. On a [draggable point](draggable-points.md), drag the point instead. |
| Mouse wheel or trackpad | Zoom about the cursor, eased. |
| `Shift` / `Ctrl` + wheel | Stretch only x / only y (needs [`equal_aspect=False`](the-plane.md#equal-aspect)). |
| Arrow keys | Pan by 10% of the plane, eased. |
| `+` / `-` | Zoom about the middle of the plane, eased. |
| `Home` / `0` | Return to the [home view](the-plane.md#setting-up-the-view). |

The mouse controls only react inside the plane's rectangle. Keys are
ignored while `Ctrl`, `Alt` or `Meta` is held, so your own shortcuts such as
`Ctrl+S` never move the view.

**Eased motion.** Wheel and key movements glide to their target instead of
jumping: each notch adds to the remaining movement, so fast scrolling feels
continuous. The easing is a critically damped spring advanced by
`draw_grid`, independent of the frame rate. Dragging is never eased; it
follows the mouse exactly.

## Configuring the controls

Pass an `InteractionConfig` (all fields optional) when creating the plane,
or assign one later:

```python
plane = pgp.Plane(interaction=pgp.InteractionConfig(zoom_per_notch=1.3))
plane.interaction = pgp.InteractionConfig(keyboard=False, smoothing=0.0)
```

| Field | Default | Meaning |
|---|---|---|
| `pan_button` | `1` | Mouse button that drags (1 left, 2 middle, 3 right; `0` disables dragging). |
| `wheel_zoom` | `True` | Whether the wheel zooms. |
| `zoom_per_notch` | `1.15` | Zoom factor per wheel notch (trackpads zoom proportionally to their scroll). |
| `invert_wheel` | `False` | Reverse the wheel direction. |
| `keyboard` | `True` | Whether the keyboard controls are active. |
| `key_pan_fraction` | `0.1` | Arrow-key pan as a fraction of the plane's size. |
| `key_zoom_factor` | `1.25` | Zoom factor per `+`/`-` press. |
| `smoothing` | `0.05` | Easing time constant in seconds (90% of a move takes about four times this). `0` moves instantly. |
| `bindings` | `KeyBindings()` | Which keys do what (below). |
| `drag_button` | `1` | Mouse button that drags [draggable points](draggable-points.md). |
| `grab_margin_px` | `6.0` | How far beyond its marker a draggable point can be grabbed. |
| `hover_cursor` | `True` | Show the hand cursor over draggable points. |

Use `smoothing=0` if your program only redraws when events arrive (eased
motion needs frames to advance).

### Key bindings

`KeyBindings` maps each action to a tuple of pygame key codes. An empty
tuple disables the action.

| Field | Default keys |
|---|---|
| `pan_left`, `pan_right`, `pan_up`, `pan_down` | the arrow keys |
| `zoom_in` | `+`, `=`, keypad `+` |
| `zoom_out` | `-`, keypad `-` |
| `reset` | `Home`, `0`, keypad `0` |

```python
from dataclasses import replace

wasd = pgp.KeyBindings(
    pan_left=(pygame.K_a,), pan_right=(pygame.K_d,),
    pan_up=(pygame.K_w,), pan_down=(pygame.K_s,),
)
plane.interaction = replace(plane.interaction, bindings=wasd)
```

## Handling your own input

`handle_event` returns `True` when the plane used an event. Use that to keep
your own handling from reacting to the same input:

```python
for event in pygame.event.get():
    if plane.handle_event(event):
        continue                       # the plane panned, zoomed or dragged a point
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
        x, y = plane.to_world(*event.pos)
        print(f"right click at ({x:.3g}, {y:.3g})")
```

To switch the pan and zoom controls off (for a fixed view, or while a
dialog is open), set `plane.interactive = False`; `handle_event` then
ignores everything except [draggable points](draggable-points.md), which
still work (set their `enabled` to `False` to stop them too). To keep the mouse but not the keys, or the reverse, use
`InteractionConfig(keyboard=False)` or `InteractionConfig(wheel_zoom=False,
pan_button=0)`.

## Moving the view from code

The [view](the-plane.md#the-view) can be changed at any time. These moves
are instant:

```python
plane.view.center = (10.0, 0.0)
plane.view.pan_pixels(-50, 0)          # content moves 50 px left
plane.view.zoom_at(400, 300, 1.5)      # zoom about a pixel
plane.set_view(x_range=(0, 1))         # fit a range
```

`set_view` and `reset_view` also cancel any eased movement still in
progress, so the user's last scroll does not continue after your jump.

To animate the view, change it a little every frame. For example, to glide
smoothly to a new centre over half a second:

```python
start, goal = plane.view.center, (5.0, 2.0)
for frame in range(30):
    t = (frame + 1) / 30
    t = t * t * (3 - 2 * t)             # ease in and out
    plane.view.center = (start[0] + (goal[0] - start[0]) * t,
                         start[1] + (goal[1] - start[1]) * t)
    plane.draw_grid(screen)
    pygame.display.flip()
```

(In a real program the frames would be part of your main loop.)
