import numpy as np
import time
from game.SpaceInvaders import SpaceInvaders
from controller.epsilon_profile import EpsilonProfile
import pandas as pd
import os
import pickle
from matplotlib import pyplot as plt


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
        self.Q = np.zeros([game.nbre_intervalle_x, game.nbre_intervalle_y, game.nbre_intervalle_x, game.nbre_intervalle_y, 2, game.na])

        self.game = game
        self.na = game.na

        # Paramètres de l'algorithme
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        # Visualisation des données (vous n'avez pas besoin de comprendre cette partie)
 

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
        q_array = np.zeros(n_episodes)
        r_array = np.zeros(n_episodes)
        # Execute N episodes 
        for episode in range(n_episodes):
            somme = 0
            # Reinitialise l'gameironnement
            state = game.reset()
            
            self.game.display = False
            #display only last 3 episodes
            if(n_episodes-episode) < 3 : 
                self.game.display = True
            # Execute K steps 
            for step in range(max_steps):
                # Selectionne une action 
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = game.step(action)
                somme += reward
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)
                
                if terminal:
                    n_steps[episode] = step + 1  
                    break

                state = next_state
            # Mets à jour la valeur du epsilon
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)

            # affiche les données d'apprentissage
            if n_episodes >= 0:
                state = game.reset()
                print("\r#> Ep. {}/{} Value {}".format(episode, n_episodes, self.Q[state][self.select_greedy_action(state)]), end =" ")
                
            q_array[episode]=np.sum(self.Q)
            r_array[episode]=somme
        
        fig1 = plt.figure("Sum of Q function values over episodes")
        plt.plot(q_array)
        fig1.suptitle('Somme des valeurs de la fonction Q en fonction de l\'épisode ', fontsize=12)
        plt.xlabel('Numéro d\'épisode', fontsize=8)
        plt.ylabel('Somme des valeurs de la fonction Q', fontsize=8)
        
        fig2 = plt.figure("Sum of rewards over episodes")
        
        plt.plot(r_array)
        fig2.suptitle('Somme des récompenses en fonction de l\'épisode ', fontsize=12)
        plt.xlabel('Numéro d\'épisode', fontsize=8)
        plt.ylabel('Somme des récompenses', fontsize=8)

        plt.show()



    def updateQ(self, state, action, reward, next_state):
        """À COMPLÉTER!
        Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent. 
        Une transition est définie comme un tuple (état, action récompense, état suivant).
        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """
        self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))

    def select_action(self, state : 'Tuple[int, int, int, int, int]'):
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

    def select_greedy_action(self, state : 'Tuple[int, int, int, int, int]'):
        """
        Cette méthode retourne l'action gourmande.
        :param state: L'état courant
        :return: L'action gourmande
        """
        mx = np.max(self.Q[state])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[state] == mx)[0])


    # sauve la variable correspondant à l'apprentissage de l'agent dans un fichier 
    def save_qfunction(self, file_name: str = "test"): 
        fichierSauvegarde = open(file_name,"wb")
        pickle.dump(self.Q, fichierSauvegarde)
    
    # charge la variable correspondant à l'apprentissage de l'agent dans self.Q 
    def load_qfunction(self,file_name: str = "test"):
        fichierSauvegarde = open(file_name,"rb")
        self.Q = pickle.load(fichierSauvegarde)