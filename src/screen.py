import os
import sys
import time
import numpy as np
# from src.recover.environment import State
import pygame

class Button:
    def __init__(self,text,width,height,pos,elevation):
        #Core attributes 
        self.pressed = False
        self.elevation = elevation
        self.dynamic_elecation = elevation
        self.original_y_pos = pos[1] #Elevation
        self.gui_font = pygame.font.Font(None, 20)
 
        # top rectangle 
        self.top_rect = pygame.Rect(pos,(width,height))
        self.top_color = '#475F77'
 
        # bottom rectangle 
        self.bottom_rect = pygame.Rect(pos,(width,height))
        self.bottom_color = '#354B5E'
        #text
        self.text = text
        self.text_surf = self.gui_font.render(text,True,'#FFFFFF')
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)

        self.clicked = False
        
    def change_text(self, newtext):
        self.text_surf = self.gui_font.render(newtext, True,'#FFFFFF')
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)
 
    def draw(self, screen):
        # elevation logic 
        self.top_rect.y = self.original_y_pos - self.dynamic_elecation
        self.text_rect.center = self.top_rect.center 
 
        self.bottom_rect.midtop = self.top_rect.midtop
        self.bottom_rect.height = self.top_rect.height + self.dynamic_elecation
 
        pygame.draw.rect(screen,self.bottom_color, self.bottom_rect)
        pygame.draw.rect(screen,self.top_color, self.top_rect)
        screen.blit(self.text_surf, self.text_rect)
        self.check_click()
 
    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()
        # print(mouse_pos)
        if self.top_rect.collidepoint(mouse_pos):
            self.top_color = '#D74B4B'
            if pygame.mouse.get_pressed()[0]:
                self.dynamic_elecation = 0
                self.pressed = True
                self.change_text(f"{self.text}")
            else:
                self.dynamic_elecation = self.elevation
                if self.pressed == True:
                    print(self.text + ' clicked')
                    self.change_text(self.text)
                    self.pressed = False
        else:
            self.dynamic_elecation = self.elevation
            self.top_color = '#475F77'

LINE_COLOR = (255, 255, 255)
LINE_WIDTH = 1
GREY = (100, 100, 100)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

class Screen(object):
    '''
    Screen class
    
    '''
    def __init__(self, state):
        self.height = state.block_dim[0]
        self.width = state.block_dim[1]  
        pygame.init()
        self.SQUARE_SIZE = state.block_size[0]
        SCREEN_SIZE = self.coord(self.width + 2, self.height + 5)
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        pygame.display.set_caption( 'ProCon-2022' ) 
        self.jump_5_button = Button('J5', self.SQUARE_SIZE, self.SQUARE_SIZE,
                                    self.coord(0, self.height + 2), 0)
        self.jump_10_button = Button('J10', self.SQUARE_SIZE, self.SQUARE_SIZE,
                                    self.coord(1, self.height + 2), 0)
        self.jump_20_button = Button('J20', self.SQUARE_SIZE, self.SQUARE_SIZE,
                                    self.coord(0, self.height + 3), 0)
        self.back_1_button = Button('B1', self.SQUARE_SIZE, self.SQUARE_SIZE,
                                    self.coord(2, self.height + 2), 0)
        self.back_5_button = Button('B5', self.SQUARE_SIZE, self.SQUARE_SIZE,
                                    self.coord(3, self.height + 2), 0)
        self.accept_button = Button('AC', self.SQUARE_SIZE, self.SQUARE_SIZE,
                                    self.coord(1, self.height + 3), 0)
        self.reject_button = Button('RJ', self.SQUARE_SIZE, self.SQUARE_SIZE,
                                    self.coord(2, self.height + 3), 0)
        self.done_button = Button('DN', self.SQUARE_SIZE, self.SQUARE_SIZE,
                                    self.coord(3, self.height + 3), 0)
        self.buttons = [self.accept_button, 
                        self.reject_button,
                        self.jump_5_button,
                        self.jump_10_button,
                        self.jump_20_button,
                        self.back_1_button,
                        self.back_5_button,
                        self.done_button]
        # grey background
        # self.screen.fill((100, 100, 100))
        (x1, y1), (x2, y2) = self.coord(0, self.height + 2), SCREEN_SIZE
        pygame.draw.rect(self.screen, GREY, (x1, y1, x2, y2))
        self.warning_font = pygame.font.SysFont('Arial', 20)
        # self.render(state)
        
    def coord(self, x, y):
        return x * self.SQUARE_SIZE, y * self.SQUARE_SIZE

    def draw_line(self, x0, y0, x1, y1):
        pygame.draw.line(self.screen, LINE_COLOR, 
                         self.coord(x0, y0), self.coord(x1, y1), 
                         LINE_WIDTH)
        
    def buttons_draw(self):
        for b in self.buttons:
            b.draw(self.screen)
            
    def render(self, state, show_button = True):
        state.save_image()
        image = pygame.image.load('output/sample.png')
        image = pygame.transform.scale(image, self.coord(self.width, self.height))
        self.screen.blit(image, self.coord(1, 1))
        self.buttons_draw()
        pygame.display.update()
    
    def get_mouse_clicked_position(self):
        events = pygame.event.get()
        if len(events) > 0:
            for event in reversed(events):
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    x = pos[1] // self.SQUARE_SIZE
                    y = pos[0] // self.SQUARE_SIZE
                    if x <= self.height and y <= self.width:
                        return (x - 1, y - 1)
                    else:
                        return None
        else:
            return None
    
    def start(self, env, state, algo):
        n_jumps = 0
        chosen_position = None
        waiting_mode = False
        recommended_block_position = 0
        action = None
        start = time.time()
        threshold = algo.threshold
        while self.done_button.pressed == False:
            # pygame.event.get()
            chosen_position = self.get_mouse_clicked_position()
            if chosen_position != None:
                print(chosen_position)
                if chosen_position[0] < state.block_dim[0] and \
                    chosen_position[1] < state.block_dim[1]:
                    x, y = chosen_position
                    if state.masked[x][y] == 0:
                        waiting_mode = False    
                    else:
                        state = env.remove(state, (x, y))
                
            if state.depth >= state.max_depth:
                n_jumps = 0
                self.accept_button.pressed = False
                waiting_mode = True
            
            if not waiting_mode:
                # try:
                actions, probs = algo.get_next_action(state, position=chosen_position)
                waiting_mode = True
                chosen_position = None
                recommended_block_position = 0
                action = actions[0]
                state = env.step(state, action)
                print('Probability: {} / {}'.format(
                    np.round(probs[recommended_block_position], 2),
                    algo.threshold))
                print('Step: {} / {}'.format(state.depth, state.max_depth))
                print('Time: %.3f' % (time.time() - start))
                # except Exception as e:
                #     print(e)
                #     waiting_mode = True
                    
            
            if n_jumps > 0:
                # try:
                if probs[0] > 0.1:
                    waiting_mode = False
                    n_jumps -= 1
                    self.render(state)
                    continue
                else:
                    n_jumps = 0
                # except Exception as e:
                #     print(e)
            else:
                text = self.warning_font.render('*', True, BLACK)
                self.screen.blit(text, self.coord(1, self.height + 1))
            
            if self.reject_button.pressed:
                # try:
                recommended_block_position += 1
                recommended_block_position %= len(actions)
                self.reject_button.pressed = False
                action = actions[recommended_block_position]
                print('Probability: {} / {}'.format(
                    np.round(probs[recommended_block_position], 2),
                    algo.threshold))
                print('Step: {} / {}'.format(state.depth, state.max_depth))
                print('Time: %.3f' % (time.time() - start))
                if state.parent is not None:
                    state = state.parent
                state = env.step(state, action)
                pygame.event.clear()
                pygame.time.delay(500)
                continue
                # except Exception as e:
                #     print(e)
            
            if self.accept_button.pressed:
                n_jumps -= 1
                self.accept_button.pressed = False
                waiting_mode = False
                pygame.event.clear()
                pygame.time.delay(500)
                continue
                
            if self.back_1_button.pressed:
                self.back_1_button.pressed = False
                if state.parent is not None:
                    state = state.parent
                pygame.event.clear()
                pygame.time.delay(500)
                algo.threshold *= 1.02
                continue
                
            if self.back_5_button.pressed:
                self.back_5_button.pressed = False
                for _ in range(6):
                    if state.parent is not None:
                        state = state.parent
                pygame.event.clear()
                algo.threshold *= 1.02
                pygame.time.delay(500)
                waiting_mode = False
                continue
                
            if self.jump_5_button.pressed:
                self.jump_5_button.pressed = False
                n_jumps += 5
                algo.threshold *= 0.99
                pygame.event.clear()
                pygame.time.delay(500)
                text = self.warning_font.render('*', True, YELLOW)
                self.screen.blit(text, self.coord(1, self.height + 1))
                continue
            
            if self.jump_10_button.pressed:
                self.jump_10_button.pressed = False
                n_jumps += 10
                algo.threshold *= 0.98
                pygame.event.clear()
                pygame.time.delay(500)
                text = self.warning_font.render('*', True, YELLOW)
                self.screen.blit(text, self.coord(1, self.height + 1))
                continue
                
            if self.jump_20_button.pressed:
                self.jump_20_button.pressed = False
                n_jumps += 20
                algo.threshold *= 0.97
                pygame.event.clear()
                pygame.time.delay(500)
                text = self.warning_font.render('*', True, YELLOW)
                self.screen.blit(text, self.coord(1, self.height + 1))
                continue
            
            self.render(state)
    
        return state