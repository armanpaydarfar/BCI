import pygame
import config


def draw_time_balls(ball_state, screen_width, screen_height, ball_radius=60, mode="single"):
    """
    Draw a time indicator ball with 4 possible states:
    - 0 = Empty (Outlined Ball)
    - 1 = White Ball (Baseline/Neutral)
    - 2 = Red Ball (Motor Imagery)
    - 3 = Blue Ball (Rest)

    :param ball_state: The current state of the ball (0-3).
    :param screen_width: Width of the screen.
    :param screen_height: Height of the screen.
    :param ball_radius: Radius of the ball(s).
    :param mode: "single" (default) for one ball, "stack" for a three-ball countdown.
    """

    # Define ball colors based on state
    color_map = {
        1: (255, 255, 255),  # White (Baseline)
        2: (255, 0, 0),  # Red (MI)
        3: (0, 0, 255)  # Blue (Rest)
    }
    
    ball_color = color_map.get(ball_state, (255, 255, 255))  # Default to white

    if mode == "single":
        # Single ball centered horizontally, positioned below the fixation cross
        ball_x = screen_width // 2
        ball_y = screen_height // 2 - ball_radius * 4  # Position below fixation cross

        if ball_state == 0:
            # Draw an outlined ball (empty state)
            pygame.draw.circle(pygame.display.get_surface(), (255, 255, 255), (ball_x, ball_y), ball_radius, 2)
        else:
            # Draw a filled ball
            pygame.draw.circle(pygame.display.get_surface(), ball_color, (ball_x, ball_y), ball_radius)

    elif mode == "stack":
        # Three vertically stacked balls for countdown
        stack_x = screen_width // 2 - ball_radius * 14  # Positioned to the left
        stack_y_start = screen_height // 2 - ball_radius * 2  # Center vertically
        spacing = ball_radius * 3  # Space between balls

        for i in range(3):
            ball_y = stack_y_start + i * spacing
            if ball_state == 0:
                pygame.draw.circle(pygame.display.get_surface(), (255, 255, 255), (stack_x, ball_y), ball_radius, 2)
            else:
                pygame.draw.circle(pygame.display.get_surface(), ball_color, (stack_x, ball_y), ball_radius)


def draw_arrow_fill(progress, screen_width, screen_height, show_threshold=True):
    ball_radius = 200  # Base measurement
    bar_width, bar_length = ball_radius * 2, ball_radius * 2
    offset = ball_radius * 2
    if config.ARM_SIDE == "Left":
        bar_x = screen_width // 2 - offset
    else:
        bar_x = screen_width // 2 + offset
    
    bar_y = screen_height // 2


    bar_outline = [
        (bar_x - bar_length // 2, bar_y - bar_width // 2),
        (bar_x + bar_length // 2, bar_y - bar_width // 2),
        (bar_x + bar_length // 2, bar_y + bar_width // 2),
        (bar_x - bar_length // 2, bar_y + bar_width // 2),
    ]
    pygame.draw.polygon(pygame.display.get_surface(), (255, 255, 255), bar_outline, 2)

    # Calculate fill length
    fill_length = int(progress * bar_length)

    filled_rect = pygame.Rect(
        bar_x - bar_length // 2, bar_y - bar_width // 2,
        fill_length, bar_width
    )
    pygame.draw.rect(pygame.display.get_surface(), (255, 0, 0), filled_rect)

    # Draw success threshold line if enabled
    if show_threshold:
        # Scale accuracy threshold within the shape boundaries
        scaled_threshold = (config.THRESHOLD_MI - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)
        scaled_threshold = max(0, min(1, scaled_threshold))  # Keep within [0,1] range

        # Compute threshold bar position using scaled threshold
        threshold_x = bar_x - bar_length // 2 + int(scaled_threshold * bar_length)

        for i in range(0, bar_width, 10):
            pygame.draw.line(
                pygame.display.get_surface(), (255, 0, 0),
                (threshold_x, bar_y - bar_width // 2 + i),
                (threshold_x, bar_y - bar_width // 2 + i + 5), 2
            )


def draw_ball_fill(progress, screen_width, screen_height, show_threshold=True):
    ball_radius = 200
    offset = ball_radius * 2
    if config.ARM_SIDE == "Left":
        ball_x = screen_width // 2 + offset
    else:
        ball_x = screen_width // 2 - offset
    ball_y = screen_height // 2

    pygame.draw.circle(pygame.display.get_surface(), (255, 255, 255), (ball_x, ball_y), ball_radius, 2)
    water_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    fill_height = int(progress * ball_radius * 2)
    water_rect = pygame.Rect(0, ball_radius * 2 - fill_height, ball_radius * 2, fill_height)
    pygame.draw.rect(water_surface, (0, 0, 255, 180), water_rect)
    mask_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (ball_radius, ball_radius), ball_radius)
    water_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pygame.display.get_surface().blit(water_surface, (ball_x - ball_radius, ball_y - ball_radius))

    # Draw success threshold line if enabled
    if show_threshold:
        # Scale accuracy threshold within the shape boundaries
        scaled_threshold = (config.THRESHOLD_REST - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)
        scaled_threshold = max(0, min(1, scaled_threshold))  # Keep within [0,1] range

        # Compute threshold position using scaled threshold
        threshold_y = ball_y + ball_radius - int(scaled_threshold * (ball_radius * 2))

        for i in range(0, ball_radius * 2, 10):
            pygame.draw.line(
                pygame.display.get_surface(), (0, 0, 255), 
                (ball_x - ball_radius + i, threshold_y), 
                (ball_x - ball_radius + i + 5, threshold_y), 2)

def draw_fixation_cross(screen_width, screen_height):
    cross_length = 40
    line_thickness = 6
    fixation_x, fixation_y = screen_width // 2, screen_height // 2

    pygame.draw.line(pygame.display.get_surface(), (255, 255, 255), 
                     (fixation_x, fixation_y - cross_length), 
                     (fixation_x, fixation_y + cross_length), 
                     line_thickness)

    pygame.draw.line(pygame.display.get_surface(), (255, 255, 255), 
                     (fixation_x - cross_length, fixation_y), 
                     (fixation_x + cross_length, fixation_y), 
                     line_thickness)


def _dashed_vertical_line(surf, color, x, y0, y1, width, dash=6, gap=4):
    y = y0
    while y < y1:
        pygame.draw.line(surf, color, (x, y), (x, min(y + dash, y1)), width)
        y += dash + gap


def _dashed_horizontal_line(surf, color, y, x0, x1, width, dash=6, gap=4):
    x = x0
    while x < x1:
        pygame.draw.line(surf, color, (x, y), (min(x + dash, x1), y), width)
        x += dash + gap


def draw_arrow_fill_modern(progress, screen_width, screen_height, show_threshold=True):
    """MI cue: same fill semantics as draw_arrow_fill; refined outline and dashed threshold."""
    ball_radius = 200
    bar_width, bar_length = ball_radius * 2, ball_radius * 2
    offset = ball_radius * 2
    if config.ARM_SIDE == "Left":
        bar_x = screen_width // 2 - offset
    else:
        bar_x = screen_width // 2 + offset
    bar_y = screen_height // 2
    surf = pygame.display.get_surface()

    outline = [
        (bar_x - bar_length // 2, bar_y - bar_width // 2),
        (bar_x + bar_length // 2, bar_y - bar_width // 2),
        (bar_x + bar_length // 2, bar_y + bar_width // 2),
        (bar_x - bar_length // 2, bar_y + bar_width // 2),
    ]
    pygame.draw.polygon(surf, (240, 240, 240), outline, width=3)

    fill_length = int(progress * bar_length)
    if fill_length > 0:
        filled_rect = pygame.Rect(
            bar_x - bar_length // 2, bar_y - bar_width // 2,
            fill_length, bar_width
        )
        pygame.draw.rect(surf, (220, 70, 70), filled_rect)

    if show_threshold:
        scaled_threshold = (config.THRESHOLD_MI - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)
        scaled_threshold = max(0, min(1, scaled_threshold))
        threshold_x = bar_x - bar_length // 2 + int(scaled_threshold * bar_length)
        _dashed_vertical_line(
            surf, (255, 140, 140), threshold_x,
            bar_y - bar_width // 2, bar_y + bar_width // 2, 2
        )


def draw_ball_fill_modern(progress, screen_width, screen_height, show_threshold=True):
    """REST cue: same fill semantics as draw_ball_fill; softer fill and dashed threshold."""
    ball_radius = 200
    offset = ball_radius * 2
    if config.ARM_SIDE == "Left":
        ball_x = screen_width // 2 + offset
    else:
        ball_x = screen_width // 2 - offset
    ball_y = screen_height // 2
    surf = pygame.display.get_surface()

    pygame.draw.circle(surf, (240, 240, 240), (ball_x, ball_y), ball_radius, 3)
    water_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    fill_height = int(progress * ball_radius * 2)
    water_rect = pygame.Rect(0, ball_radius * 2 - fill_height, ball_radius * 2, fill_height)
    pygame.draw.rect(water_surface, (80, 130, 255, 200), water_rect)
    mask_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (ball_radius, ball_radius), ball_radius)
    water_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    surf.blit(water_surface, (ball_x - ball_radius, ball_y - ball_radius))

    if show_threshold:
        scaled_threshold = (config.THRESHOLD_REST - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)
        scaled_threshold = max(0, min(1, scaled_threshold))
        threshold_y = ball_y + ball_radius - int(scaled_threshold * (ball_radius * 2))
        _dashed_horizontal_line(
            surf, (160, 190, 255), threshold_y,
            ball_x - ball_radius, ball_x + ball_radius, 2
        )


def draw_fixation_cross_modern(screen_width, screen_height):
    cross_length = 32
    line_thickness = 4
    fixation_x, fixation_y = screen_width // 2, screen_height // 2
    c = (220, 220, 220)
    surf = pygame.display.get_surface()
    pygame.draw.line(
        surf, c,
        (fixation_x, fixation_y - cross_length),
        (fixation_x, fixation_y + cross_length),
        line_thickness
    )
    pygame.draw.line(
        surf, c,
        (fixation_x - cross_length, fixation_y),
        (fixation_x + cross_length, fixation_y),
        line_thickness
    )


def draw_time_balls_modern(ball_state, screen_width, screen_height, ball_radius=48, mode="single"):
    """Smaller time ball(s) for modern style."""
    color_map = {
        1: (245, 245, 245),
        2: (230, 80, 80),
        3: (90, 120, 255),
    }
    ball_color = color_map.get(ball_state, (245, 245, 245))
    surf = pygame.display.get_surface()

    if mode == "single":
        ball_x = screen_width // 2
        ball_y = screen_height // 2 - ball_radius * 4
        if ball_state == 0:
            pygame.draw.circle(surf, (200, 200, 200), (ball_x, ball_y), ball_radius, 2)
        else:
            pygame.draw.circle(surf, ball_color, (ball_x, ball_y), ball_radius)
    elif mode == "stack":
        stack_x = screen_width // 2 - ball_radius * 14
        stack_y_start = screen_height // 2 - ball_radius * 2
        spacing = ball_radius * 3
        for i in range(3):
            ball_y = stack_y_start + i * spacing
            if ball_state == 0:
                pygame.draw.circle(surf, (200, 200, 200), (stack_x, ball_y), ball_radius, 2)
            else:
                pygame.draw.circle(surf, ball_color, (stack_x, ball_y), ball_radius)


def draw_class_feedback_cues(
    style,
    mode,
    mi_fill,
    rest_fill,
    screen_width,
    screen_height,
    time_ball_state,
    accumulation=None,
):
    """
    Draw MI/REST feedback shapes for one frame. Preserves classic ordering and semantics.

    style: "classic" | "modern"
    mode: 0 = MI trial, 1 = REST trial
    accumulation: optional 0..1 for bottom progress bar (modern only)
    """
    st = (style or "classic").strip().lower()
    if st == "modern":
        if mode == 0:
            draw_arrow_fill_modern(mi_fill, screen_width, screen_height)
            draw_fixation_cross_modern(screen_width, screen_height)
            draw_ball_fill_modern(rest_fill, screen_width, screen_height)
            draw_time_balls_modern(time_ball_state, screen_width, screen_height)
        else:
            draw_ball_fill_modern(rest_fill, screen_width, screen_height)
            draw_fixation_cross_modern(screen_width, screen_height)
            draw_arrow_fill_modern(mi_fill, screen_width, screen_height)
            draw_time_balls_modern(time_ball_state, screen_width, screen_height)
        if accumulation is not None:
            draw_progress_bar(
                float(accumulation), screen_width, screen_height,
                color=(200, 200, 200), height_ratio=0.04
            )
        return

    if mode == 0:
        draw_arrow_fill(mi_fill, screen_width, screen_height)
        draw_fixation_cross(screen_width, screen_height)
        draw_ball_fill(rest_fill, screen_width, screen_height)
        draw_time_balls(time_ball_state, screen_width, screen_height)
    else:
        draw_ball_fill(rest_fill, screen_width, screen_height)
        draw_fixation_cross(screen_width, screen_height)
        draw_arrow_fill(mi_fill, screen_width, screen_height)
        draw_time_balls(time_ball_state, screen_width, screen_height)


def draw_class_fixation_idle(style, screen_width, screen_height):
    """Fixation period: zero-fill cues."""
    st = (style or "classic").strip().lower()
    if st == "modern":
        draw_fixation_cross_modern(screen_width, screen_height)
        draw_ball_fill_modern(0, screen_width, screen_height, show_threshold=False)
        draw_arrow_fill_modern(0, screen_width, screen_height, show_threshold=False)
        draw_time_balls_modern(0, screen_width, screen_height)
    else:
        draw_fixation_cross(screen_width, screen_height)
        draw_ball_fill(0, screen_width, screen_height)
        draw_arrow_fill(0, screen_width, screen_height)
        draw_time_balls(0, screen_width, screen_height)


def draw_progress_bar(progress, screen_width, screen_height, color=config.green, height_ratio=0.05):
    """
    Draws a horizontal progress bar at the bottom of the screen.

    Args:
        progress (float): Value between 0.0 and 1.0 representing completion.
        screen_width (int): Width of the display.
        screen_height (int): Height of the display.
        color (tuple): RGB color of the fill.
        height_ratio (float): Bar height as a fraction of screen height.
    """
    progress = max(0.0, min(1.0, progress))  # clamp

    # Bar geometry
    bar_width = int(screen_width * 0.6)
    bar_height = int(screen_height * height_ratio)
    bar_x = (screen_width - bar_width) // 2
    bar_y = int(screen_height * 0.8)  # 80% down the screen

    # Border
    pygame.draw.rect(
        pygame.display.get_surface(), config.white,
        (bar_x, bar_y, bar_width, bar_height), width=2
    )

    # Fill
    fill_width = int(bar_width * progress)
    if fill_width > 0:
        pygame.draw.rect(
            pygame.display.get_surface(), color,
            (bar_x + 2, bar_y + 2, max(0, fill_width - 4), bar_height - 4)
        )




'''
def draw_arrow_fill(progress, screen_width, screen_height):
    arrow_width, arrow_length, tip_length = 80, 200, 40
    arrow_x, arrow_y = (6 * screen_width) // 8, screen_height // 2

    arrow_outline = [
        (arrow_x - arrow_length // 2, arrow_y - arrow_width // 2),
        (arrow_x + arrow_length // 2 - tip_length, arrow_y - arrow_width // 2),
        (arrow_x + arrow_length // 2, arrow_y),
        (arrow_x + arrow_length // 2 - tip_length, arrow_y + arrow_width // 2),
        (arrow_x - arrow_length // 2, arrow_y + arrow_width // 2),
    ]
    pygame.draw.polygon(pygame.display.get_surface(), (255, 255, 255), arrow_outline, 2)
    filled_rect = pygame.Rect(
        arrow_x - arrow_length // 2, arrow_y - arrow_width // 2,
        int(progress * (arrow_length - tip_length)),
        arrow_width
    )
    pygame.draw.rect(pygame.display.get_surface(), (255, 0, 0), filled_rect)

def draw_ball_fill(progress, screen_width, screen_height):
    ball_radius = 100
    ball_x, ball_y = screen_width // 2, screen_height // 4

    pygame.draw.circle(pygame.display.get_surface(), (255, 255, 255), (ball_x, ball_y), ball_radius, 2)
    water_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    fill_height = int(progress * ball_radius * 2)
    water_rect = pygame.Rect(0, ball_radius * 2 - fill_height, ball_radius * 2, fill_height)
    pygame.draw.rect(water_surface, (0, 0, 255, 180), water_rect)
    mask_surface = pygame.Surface((ball_radius * 2, ball_radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (ball_radius, ball_radius), ball_radius)
    water_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pygame.display.get_surface().blit(water_surface, (ball_x - ball_radius, ball_y - ball_radius))

def draw_fixation_cross(screen_width, screen_height):
    cross_length = 40
    line_thickness = 6
    fixation_x, fixation_y = screen_width // 2, screen_height // 2
    pygame.draw.line(pygame.display.get_surface(), (255, 255, 255), (fixation_x, fixation_y - cross_length), (fixation_x, fixation_y + cross_length), line_thickness)
    pygame.draw.line(pygame.display.get_surface(), (255, 255, 255), (fixation_x - cross_length, fixation_y), (fixation_x + cross_length, fixation_y), line_thickness)
'''