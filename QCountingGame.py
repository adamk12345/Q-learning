import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lim =10
choicelim =2
#choice =1 to 10
num_of_games = 2000

class Game:
    board =None
    board_lim=0
    
    def __init__(self, board_lim=lim):
        self.board_lim = board_lim
        self.reset()
        
    def reset(self):
        self.board = random.randint(1,choicelim) #this is opponent go first
       
        
    def opponentplay(self, opponentchoice):
        #opponent makes their move, game ends or not
        start_board = self.board + opponentchoice
        game_over = start_board >= self.board_lim
        if game_over:
            self.board = start_board
            return game_over
        else:
            self.board = start_board
            return game_over    
        
    def ourplay(self, choice):
        # returns a tuple : (reward, game_over?)
        new_board = self.board + choice
        if new_board < self.board_lim:#if our choice stays in bounds, set board and let opponent go
            self.board = new_board
            game_over = False
            x = random.randint(1,choicelim)
            opp_game_over = self.opponentplay(x)
            if opp_game_over:
                return (100, game_over, opp_game_over)
            else:
                return (0, game_over, opp_game_over) #continue on
        else:
            return (-100, True, False) #second one, did i end game, third did they end game
        #need to build opponent action into our reward here, careful with how updates state
        
        
   
game = Game()

epsilon = 0.1
gamma = 1  #discount rate          
alpha = 0.7

q_table = pd.DataFrame(0, index=np.arange(1,choicelim+1,1), columns=np.arange(lim))
r_list = []

for g in range(num_of_games):
    game_over = False
    opp_game_over=False
    game.reset() #this now makes opponent go first
    total_reward = 0
    while not game_over and not opp_game_over:
        state=game.board
        if random.random()<epsilon:
            choice = random.randint(1,choicelim)
        else:
            choice = q_table[state].idxmax() #issue with this is picks lowest index when tied -> us picking low numbers+tending to win more when have less scenarios
        reward, game_over, opp_game_over = game.ourplay(choice) #this does our move+opponents
        total_reward += reward
        if game_over or opp_game_over: #if weve lost or opponent has, q(next state) used to update will be 0 as be in terminal state
            next_state_max_q_val = 0
        else: #next state is where opponent leaves us   
            next_state =game.board # this should always be in bounds
            next_state_max_q_val = q_table[next_state].max()
           
        q_table.loc[choice,state] = q_table.loc[choice,state]+alpha*(reward + gamma * (next_state_max_q_val -q_table.loc[choice,state]))
    r_list.append(total_reward)  
    
#test q optimal
epsilon=0    
r_list = []

for g in range(2000):
    game_over = False
    opp_game_over=False
    game.reset() #this now makes opponent go first
    total_reward = 0
    while not game_over and not opp_game_over:
        state=game.board
        if random.random()<epsilon:
            choice = random.randint(1,choicelim)
        else:
            choice = q_table[state].idxmax()
        reward, game_over, opp_game_over = game.ourplay(choice) #this does our move+opponents
        total_reward += reward
        if game_over or opp_game_over: #if weve lost or opponent has, q(next state) used to update will be 0 as be in terminal state
            next_state_max_q_val = 0
        else: #next state is where opponent leaves us   
            next_state =game.board # this should always be in bounds
            next_state_max_q_val = q_table[next_state].max()
           
        q_table.loc[choice,state] = q_table.loc[choice,state]+alpha*(reward + gamma * (next_state_max_q_val -q_table.loc[choice,state]))
    r_list.append(total_reward)     

    
plt.figure(figsize=(14,7))
plt.plot(range(2000),r_list)
plt.xlabel('Games played')
plt.ylabel('Reward')
plt.show()    
