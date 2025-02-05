import pygame
import numpy as np
import random


class LeakyIntegrator:
    def __init__(self, alpha=0.95):
        """
        Initializes the leaky integrator.
        
        :param alpha: Leak factor (0 < alpha < 1), where higher values retain past values longer.
        """
        self.alpha = alpha
        self.accumulated_probability = 0.5  # Initial probability

    def update(self, new_probability):
        """
        Updates the accumulated probability using a leaky integration method.

        :param new_probability: The new probability input from the classifier.
        :return: The updated accumulated probability.
        """
        self.accumulated_probability = self.alpha * self.accumulated_probability + (1 - self.alpha) * new_probability
        return self.accumulated_probability



def generate_trial_sequence(total_trials=30, max_repeats=3):
    trials = [0] * (total_trials // 2) + [1] * (total_trials // 2)
    random.shuffle(trials)
    fixed_trials = []
    for trial in trials:
        if len(fixed_trials) >= max_repeats and all(t == trial for t in fixed_trials[-max_repeats:]):
            alternatives = [t for t in set([0, 1]) if t != trial]
            random.shuffle(alternatives)
            trial = alternatives[0]
        fixed_trials.append(trial)
    return fixed_trials



def display_multiple_messages_with_udp(messages, colors, offsets, duration=13, udp_messages=None, udp_socket=None, udp_ip=None, udp_port=None):
    font = pygame.font.SysFont(None, 72)
    end_time = pygame.time.get_ticks() + duration * 1000

    udp_sent = False
    while pygame.time.get_ticks() < end_time:
        pygame.display.get_surface().fill((0, 0, 0))
        for i, text in enumerate(messages):
            message = font.render(text, True, colors[i])
            pygame.display.get_surface().blit(
                message,
                (pygame.display.get_surface().get_width() // 2 - message.get_width() // 2,
                 pygame.display.get_surface().get_height() // 2 - message.get_height() // 2 + offsets[i])
            )
        pygame.display.flip()
        if udp_messages and not udp_sent:
            for msg in udp_messages:
                udp_socket.sendto(msg.encode('utf-8'), (udp_ip, udp_port))
                print(f"Sent UDP message to {udp_ip}:{udp_port}: {msg}")
            udp_sent = True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        pygame.time.Clock().tick(60)