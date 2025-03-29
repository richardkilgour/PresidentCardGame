import pygame

def blit_text(surface, text, pos, font, color=pygame.Color('black')):
    max_width, max_height = surface.get_size()
    x, y = pos
    word_surface = font.render(text, 0, color)
    word_width, word_height = word_surface.get_size()
    for line in text.splitlines():
        word_surface = font.render(line, 0, color)
        word_width, word_height = word_surface.get_size()
        surface.blit(word_surface, (x, y))
        y += word_height  # Start on new row.

class button_label(pygame.sprite.Sprite):
    def __init__(self, text, x_pos=0, y_pos=0, width=120, height=40, font = 'Arial', font_size=24, border = 2):
        pygame.sprite.Sprite.__init__(self)
        self.x = x_pos
        self.y = y_pos
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont(font, font_size)
        self.highlighted = False
        self.text = text
        self.border = border
        self.render()

    def set_text(self, text):
        # todo:abstract for all the GUI elements
        self.text = text
        self.render()

    def render(self):
        self.image = pygame.Surface((self.width, self.height))
        if self.highlighted:
            self.image.fill(pygame.color.THECOLORS['yellow'])
        else:
            self.image.fill(pygame.color.THECOLORS['black'])

        pygame.draw.rect(self.image, pygame.color.THECOLORS['white'],
                         (self.border, self.border, self.width - 2 * self.border, self.height - 2 * self.border))
        img = self.font.render(self.text, 1, (0, 0, 0))
        self.image.blit(img, [0, 0])
        self.rect = self.image.get_rect()
        self.rect.x = self.x
        self.rect.y = self.y

    def highlight(self, highlight):
        if self.highlighted != highlight:
            self.highlighted = highlight
            self.render()


class StatBox(pygame.sprite.Sprite):
    def __init__(self, x_pos=0, y_pos=0, width=250, height=150):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((width, height))
        self.image.fill(pygame.color.THECOLORS['aquamarine3'])
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.kingdoms = 0
        self.principalities = 0
        self.citizenships = 0
        self.assholeocity = 0
        self.total_games = 0
        self.king_run = 0
        self.ass_run = 0
        self.font = pygame.font.SysFont("Arial", 24)
        blit_text(self.image, f'Kingdoms: {self.kingdoms} Best run: {self.king_run}\nPrincipalities: {self.principalities}\nAss: {self.assholeocity}  Shitist run: {self.ass_run}\n: ', (0,0), self.font)
        self.rect = self.image.get_rect()
        self.rect.x = self.x_pos
        self.rect.y = self.y_pos

    def set_text(self, text):
        # todo:abstract for all the GUI elements
        self.text = text
        text = self.font.render(text, 1, (0, 0, 0))
        self.image = pygame.Surface((120, 50))
        self.image.fill(pygame.color.THECOLORS['white'])
        self.image.blit(text, [0, 0])
        self.rect = self.image.get_rect()
        self.rect.x = self.x_pos
        self.rect.y = self.y_pos


class PlayerNameLabel(button_label):
    def __init__(self, text, x_pos=0, y_pos=0):
        button_label.__init__(self, text, x_pos, y_pos)


# TODO: Make a generic abstract button
class PassButton(button_label):
    def __init__(self, x_pos, y_pos, width = 120, height = 40):
        button_label.__init__(self, 'Pass', x_pos, y_pos)



