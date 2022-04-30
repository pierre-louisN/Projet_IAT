import sys
import time
from controller.QAgent import QAgent
from game.SpaceInvaders import SpaceInvaders
import os 

from controller.epsilon_profile import EpsilonProfile

# parser = argparse.ArgumentParser(description='Maze parameters')
# parser.add_argument('--algo', type=str, default="random", metavar='a', help='algorithm to use (default: 7)')
# parser.add_argument('--width', type=int, default=7, metavar='w', help='width of the maze (default: 7)')
# parser.add_argument('--height', type=int, default=7, metavar='h', help='height of the maze (default: 7)')
# parser.add_argument('--shortest_path', type=int, default=14, metavar='p', help='shortest distance between starting point and goal point (default: 14)')
# args = parser.parse_args()

# test once by taking greedy actions based on Q values
def test_game(env: SpaceInvaders, agent: QAgent, max_steps: int, nepisodes : int = 1, speed: float = 0., same = True, display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    for _ in range(nepisodes):
        state = env.reset() 
        if display:
            env.render()

        while True :
            action = agent.select_greedy_action(state)
            next_state, reward, terminal = env.step(action)

            if display:
                time.sleep(speed)
                env.render()

            sum_rewards += reward
            if terminal:
                n_steps = step+1  # number of steps taken
                game.game_over(game)
                break
            state = next_state
    return n_steps, sum_rewards


def main(testing):
    # 
    env =SpaceInvaders() 
    
        
    n_episodes = 1000
    max_steps = 5000
    gamma = 1.
    alpha = 0.2
    eps_profile = EpsilonProfile(1.0, 0.1)

    agent = QAgent(env, eps_profile, gamma, alpha)
    # TRAINING  
    if (testing == "0") :  
         # TRAINING  
        agent.learn(env, n_episodes, max_steps)
        fileName = "test"
        agent.saveQToFile(os.path.join("LearnedQ", fileName))
    else : 
        # TESTING
        agent.loadQFromFile(os.path.join(os.path.abspath("LearnedQ"),"test.npy"))
        test_game(env, agent, max_steps, speed=0.5, display=False)
    

if __name__ == '__main__':
    """ Usage : python main.py [ARGS]
    - First argument (str) : the name of the agent (i.e. 'random', 'vi', 'qlearning', 'dqn')
    - Second argument (int) : the maze hight
    - Third argument (int) : the maze width
    """   
    testing = sys.argv[1]
    main(testing)