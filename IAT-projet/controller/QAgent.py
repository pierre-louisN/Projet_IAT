import numpy as np

from game.SpaceInvaders import SpaceInvaders
from controller.epsilon_profile import EpsilonProfile
import pandas as pd

class AgentInterface:
    """ 
    L'interface requise par tous les agents.
    """

    def select_action(self, state):
        """ 
        Select an action given the current policy and a state
        """
        pass

    def select_greedy_action(self, state):
        return self.select_action(state)



class QAgent(AgentInterface):
    """ 
    Cette classe d'agent représente un agent utilisant la méthode du Q-learning 
    pour mettre à jour sa politique d'action.
    """

    def __init__(self, game : SpaceInvaders,  eps_profile: EpsilonProfile, gamma: float, alpha: float):
        """A LIRE
        Ce constructeur initialise une nouvelle instance de la classe QAgent.
        Il doit stocker les différents paramètres nécessaires au fonctionnement de l'algorithme et initialiser la 
        fonction de valeur d'action, notée Q.
        :param eps_profile: Le profil du paramètre d'exploration epsilon 
        :type eps_profile: EpsilonProfile
        
        :param gamma: Le discount factor 
        :type gamma: float
        
        :param alpha: Le learning rate 
        :type alpha: float
        - Visualisation des données
        :attribut gameValues: la fonction de valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        :type gameValues: data frame pandas
        :penser à bien stocker aussi la taille du labyrinthe (nx,ny)
        :attribut qvalues: la Q-valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        
        """
        # Initialise la fonction de valeur Q
        self.Q = np.zeros([game.nbre_intervalle_x, game.nbre_intervalle_y, game.nbre_intervalle_x, game.na])

        self.game = game
        self.na = game.na

        # Paramètres de l'algorithme
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        # Visualisation des données (vous n'avez pas besoin de comprendre cette partie)
        self.qvalues = pd.DataFrame(data={'episode': [], 'value': []})
        self.values = pd.DataFrame(data={'nx': [game.nbre_intervalle_x], 'ny': [game.nbre_intervalle_y],'x2': [game.nbre_intervalle_x]})

    def learn(self, game, n_episodes, max_steps):
        """Cette méthode exécute l'algorithme de q-learning. 
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parallèle avec le cours.
        :param game: L'gameironnement 
        :type game: gym.gameselect_action
        :param num_episodes: Le nombre d'épisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'étape par épisode
        :type max_num_steps: int
        # Visualisation des données
        Elle doit proposer l'option de stockage de (i) la fonction de valeur & (ii) la Q-valeur 
        dans un fichier de log
        """
        n_steps = np.zeros(n_episodes) + max_steps
        
        # Execute N episodes 
        for episode in range(n_episodes):
            # Reinitialise l'gameironnement
            state = game.reset()
            # Execute K steps 
            for step in range(max_steps):
                # Selectionne une action 
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = game.step(action)
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)
                
                if terminal:
                    n_steps[episode] = step + 1  
                    break

                state = next_state
            # Mets à jour la valeur du epsilon
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)

            # Sauvegarde et affiche les données d'apprentissage
            if n_episodes >= 0:
                state = game.reset()
                print("\r#> Ep. {}/{} Value {}".format(episode, n_episodes, self.Q[state][self.select_greedy_action(state)]), end =" ")
                # Q : tableau à 4 dimensions (tuple de 3 avec un de plus pour les actions)
                self.save_log(game, episode)

        # self.values.to_csv('partie_3/visualisation/logV.csv')
        # self.qvalues.to_csv('partie_3/visualisation/logQ.csv')

    def updateQ(self, state, action, reward, next_state):
        """À COMPLÉTER!
        Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent. 
        Une transition est définie comme un tuple (état, action récompense, état suivant).
        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """
        #print("LES ETATS :", state)


        self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))

    def select_action(self, state : 'Tuple[int, int, int]'):
        """À COMPLÉTER!
        Cette méthode retourne une action échantilloner selon le processus d'exploration (ici epsilon-greedy).
        :param state: L'état courant
        :return: L'action 
        """
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.na)      # random action
        else:
            a = self.select_greedy_action(state)
        return a

    def select_greedy_action(self, state : 'Tuple[int, int, int]'):
        """
        Cette méthode retourne l'action gourmande.
        :param state: L'état courant
        :return: L'action gourmande
        """
        mx = np.max(self.Q[state])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[state] == mx)[0])

    def save_log(self, game, episode):
        """Sauvegarde les données d'apprentissage.
        :warning: Vous n'avez pas besoin de comprendre cette méthode
        """
        state = game.reset()
        # Construit la fonction de valeur d'état associée à Q
        V = np.zeros((int(self.game.nbre_intervalle_x), int(self.game.nbre_intervalle_y),int(self.game.nbre_intervalle_x)))
        for state in self.game.getStates():
            val = self.Q[state][self.select_action(state)]
            V[state] = val

        self.qvalues = self.qvalues.append({'episode': episode, 'value': self.Q[state][self.select_greedy_action(state)]}, ignore_index=True)
        self.values = self.values.append({'episode': episode, 'value': np.reshape(V,(1, self.game.nbre_intervalle_y*self.game.nbre_intervalle_x*self.game.nbre_intervalle_x))[0]},ignore_index=True)