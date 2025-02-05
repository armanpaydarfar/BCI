import pygame
import config


def draw_time_balls(elapsed_time, next_trial_mode, screen_width, screen_height, ball_radius=30, mode="single"):
    """
    Draw a time indicator ball. Supports two modes:
    - "single" (default): Displays a single ball that is either filled or outlined.
    - "stack": Displays three vertically stacked balls as a countdown.

    :param elapsed_time: Time elapsed during the countdown (in milliseconds).
    :param next_trial_mode: The mode of the next trial (0 for MI, 1 for rest).
    :param screen_width: Width of the screen.
    :param screen_height: Height of the screen.
    :param ball_radius: Radius of the ball(s).
    :param mode: "single" (default) for one ball, "stack" for a three-ball countdown.
    """
    # Set color based on next trial mode
    if next_trial_mode == 0:  # Right-hand motor imagery
        ball_color = (255, 0, 0)  # Red
    elif next_trial_mode == 1:  # Rest
        ball_color = (0, 0, 255)  # Blue
    else:
        ball_color = (255, 255, 255)  # Default white (if mode is undefined)

    if mode == "single":
        # Single ball centered horizontally, positioned below the fixation cross
        ball_x = screen_width // 2
        ball_y = screen_height // 2 - ball_radius * 4  # Position below fixation cross

        # Fill the ball at beginning of 3 second countdown
        if elapsed_time > 100:  
            pygame.draw.circle(pygame.display.get_surface(), ball_color, (ball_x, ball_y), ball_radius)
        else:  # Outline before the trial starts
            pygame.draw.circle(pygame.display.get_surface(), (255, 255, 255), (ball_x, ball_y), ball_radius, 2)

    elif mode == "stack":
        # Three vertically stacked balls for countdown
        stack_x = screen_width // 2 - ball_radius * 14  # Positioned to the left of the main shape
        stack_y_start = screen_height // 2 - ball_radius * 2  # Center vertically
        spacing = ball_radius * 3  # Space between balls

        # Draw the balls in a countdown format
        for i in range(3):
            ball_y = stack_y_start + i * spacing
            if elapsed_time >= i * 1000:  # Fill at 0, 1, and 2 seconds
                pygame.draw.circle(pygame.display.get_surface(), ball_color, (stack_x, ball_y), ball_radius)
            else:  # Empty outline
                pygame.draw.circle(pygame.display.get_surface(), (255, 255, 255), (stack_x, ball_y), ball_radius, 2)



def draw_arrow_fill(progress, screen_width, screen_height, show_threshold=True):
    ball_radius = 120  # Base measurement
    bar_width, bar_length = ball_radius * 2, ball_radius * 2
    bar_x, bar_y = screen_width // 2 + ball_radius * 2, screen_height // 2  # Positioned to the right

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
        scaled_threshold = (config.ACCURACY_THRESHOLD - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)
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
    ball_radius = 120
    ball_x, ball_y = screen_width // 2 - ball_radius*2, screen_height // 2  # Superimposed with fixation cross

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
        scaled_threshold = (config.ACCURACY_THRESHOLD - config.SHAPE_MIN) / (config.SHAPE_MAX - config.SHAPE_MIN)
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