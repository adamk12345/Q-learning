import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lim =10 #this is the target we dont want to hit
choicelim =2 # pick a number between 1 and choicelim
num_of_games = 2000
num_of_test_games = 2000 # number of games we will test our code on

class Game:
    # this class holds the functions used to run the game
    board =None
    board_lim=0
    
    def __init__(self, board_lim=lim):
        self.board_lim = board_lim
        self.reset()
        
    def reset(self):
        self.board = random.randint(1,choicelim) # we reset each game to opponents random first choice
       
        
    def opponentplay(self, opponentchoice):
        # opponent makes their move, game ends or not
        start_board = self.board + opponentchoice
        game_over = start_board >= self.board_lim
        if game_over:
            self.board = start_board
            return game_over
        else:
            self.board = start_board
            return game_over    
        
    def ourplay(self, choice):
        # we make our move. returns a tuple : (reward, game_over?)
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
            options = list(q_table.loc[q_table[state]==q_table[state].max()].T.columns) #lists the best moves according to q table
            choice = random.sample(options,1)[0] # randomly tiebreaks from these best moves
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
r_list = []

for g in range(num_of_test_games):
    game_over = False
    opp_game_over=False
    game.reset() #this now makes opponent go first
    total_reward = 0
    while not game_over and not opp_game_over:
        state=game.board
        choice = q_table[state].idxmax()
        reward, game_over, opp_game_over = game.ourplay(choice) #this does our move+opponents
        total_reward += reward
        if not game_over and not opp_game_over:  
            next_state =game.board # this should always be in bounds
           
    r_list.append(total_reward)     

    
plt.figure(figsize=(14,7))
plt.plot(range(num_of_test_games),r_list)
plt.xlabel('Games played')
plt.ylabel('Reward')
plt.show()    
