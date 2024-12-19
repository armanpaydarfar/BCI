import pygame

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
