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