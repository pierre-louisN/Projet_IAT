import sys
import time
from controller.QAgent import QAgent
from game.SpaceInvaders import SpaceInvaders
import os 
import pickle
from controller.epsilon_profile import EpsilonProfile


# test once by taking greedy actions based on Q values
def test_game(env: SpaceInvaders, agent: QAgent, max_steps: int, nepisodes : int = 1, speed: float = 0., same = True, display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    step = 0
    for _ in range(nepisodes):
        state = env.reset() 
        if display:
            env.render()

        while True :
            action = agent.select_greedy_action(state)
            next_state, reward, terminal = env.step(action)
            step +=1
            if display:
                time.sleep(speed)
                env.render()

            sum_rewards += reward
            if terminal:
                n_steps = step+1  # number of steps taken
                env.game_over()
                break
            state = next_state
    return n_steps, sum_rewards


def main(testing):
     
    env =SpaceInvaders() 
    
    #Hyperparamètre   
    n_episodes = 1000
    max_steps = 1000
    gamma = 0.95
    alpha = 1
    eps_profile = EpsilonProfile(1.0, 0.1)

    agent = QAgent(env, eps_profile, gamma, alpha)
    if (testing == "0") :  
         # TRAINING  
        agent.learn(env, n_episodes, max_steps)
        fileName = "test"
        agent.save_qfunction()
    else : 
        # TESTING
        agent.load_qfunction()
        test_game(env, agent, max_steps, speed=0.0001, display=True)
    

if __name__ == '__main__':
    """ Usage : python main.py [ARGS]
    - First argument (str) : the name of the agent (i.e. 'random', 'vi', 'qlearning', 'dqn')
    - Second argument (int) : the maze hight
    - Third argument (int) : the maze width
    """   
    testing = sys.argv[1]
    main(testing)