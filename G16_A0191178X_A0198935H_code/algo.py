import numpy as np
from scipy.stats import beta

class EpsilonGreedy():

    def __init__(self, epsilon, n_arms):
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.algo = 'EpsilonGreedy (ε=' + str(epsilon) + ')'

    def choose_arm(self, n_trails, user_feat, pool_idx, art_feat):
        """
        
        Return chosen arm relative to the pool
        
        Arguments:
            n_trails {int} -- number of trails
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
    
        """

        if np.random.rand() > self.epsilon:
            return np.argmax(self.values[pool_idx])
        else:
            return np.random.randint(low=0, high=len(pool_idx))

    def update(self, chosen_arm, reward, user_feat, pool_idx, art_feat):
        """
        
        Update parameters of the algo
        
        Arguments:
            chosen_arm {int} -- chosen article index relative to the pool
            reward {int} -- binary, user click is 1, not click is 0
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
        """

        a = pool_idx[chosen_arm]
        self.counts[a] += 1
        n = self.counts[a]
        self.values[a] = ((n-1)/float(n))*self.values[a] + (1/float(n))*reward

class EpsilonDecay():

    def __init__(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.algo = 'EpsilonDecay'

    def choose_arm(self, n_trails, user_feat, pool_idx, art_feat):
        """
        
        Return chosen arm relative to the pool
        
        Arguments:
            n_trails {int} -- number of trails
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
    
        """

        if np.random.rand() > 1/(sum(self.counts)/len(self.counts)+1):
            return np.argmax(self.values[pool_idx])
        else:
            return np.random.randint(low=0, high=len(pool_idx))

    def update(self, chosen_arm, reward, user_feat, pool_idx, art_feat):
        """
        
        Update parameters of the algo
        
        Arguments:
            chosen_arm {int} -- chosen article index relative to the pool
            reward {int} -- binary, user click is 1, not click is 0
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
        """

        a = pool_idx[chosen_arm]
        self.counts[a] += 1
        n = self.counts[a]
        self.values[a] = ((n-1)/float(n))*self.values[a] + (1/float(n))*reward

class AnnealingSoftmax():

    def __init__(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.algo = 'AnnealingSoftmax'

    def choose_arm(self, n_trails, user_feat, pool_idx, art_feat):
        """
        
        Return chosen arm relative to the pool
        
        Arguments:
            n_trails {int} -- number of trails
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
    
        """

        temperature = 1/(1+np.log(sum(self.counts[pool_idx])+0.000001))
        z=sum([np.exp(v/temperature) for v in self.values[pool_idx]])
        probs=[np.exp(v/temperature)/z for v in self.values[pool_idx]]
        return np.random.choice(len(pool_idx), p=probs)

    def update(self, chosen_arm, reward, user_feat, pool_idx, art_feat):
        """
        
        Update parameters of the algo
        
        Arguments:
            chosen_arm {int} -- chosen article index relative to the pool
            reward {int} -- binary, user click is 1, not click is 0
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
        """

        a = pool_idx[chosen_arm]
        self.counts[a] += 1
        n = self.counts[a]
        self.values[a] = ((n-1)/float(n))*self.values[a] + (1/float(n))*reward

class UCB1():

    def __init__(self, n_arms, alpha):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.alpha = alpha
        self.algo = 'UCB1 (α=' + str(alpha) + ')'

    def choose_arm(self, n_trails, user_feat, pool_idx, art_feat):
        """
        
        Return chosen arm relative to the pool
        
        Arguments:
            n_trails {int} -- number of trails
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
    
        """

        ucb_values = self.values[pool_idx] + \
                    np.sqrt(self.alpha * np.log(n_trails + 1) / self.counts[pool_idx])
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward, user_feat, pool_idx, art_feat):
        """
        
        Update parameters of the algo
        
        Arguments:
            chosen_arm {int} -- chosen article index relative to the pool
            reward {int} -- binary, user click is 1, not click is 0
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
        """

        a = pool_idx[chosen_arm]
        self.counts[a] += 1
        n = self.counts[a]
        self.values[a] = ((n-1)/float(n))*self.values[a] + (1/float(n))*reward

class BayesUCB():

    def __init__(self, n_arms, stdnum=3, init_alpha=1, init_beta=1):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.alphas = np.array([init_alpha] * n_arms)
        self.betas = np.array([init_beta] * n_arms)
        self.stdnum = stdnum
        self.algo = 'Baysian UCB'

    def choose_arm(self, n_trails, user_feat, pool_idx, art_feat):
        """
        
        Return chosen arm relative to the pool
        
        Arguments:
            n_trails {int} -- number of trails
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
    
        """

        pool_alphas = self.alphas[pool_idx]
        pool_betas = self.betas[pool_idx]

        best_arm = max(
                range(len(pool_idx)),
                key=lambda x: pool_alphas[x] / float(pool_alphas[x] + pool_betas[x]) + \
                    beta.std(
                        pool_alphas[x], pool_betas[x]
                    ) * self.stdnum
            )
        return best_arm

    def update(self, chosen_arm, reward, user_feat, pool_idx, art_feat):
        """
        
        Update parameters of the algo
        
        Arguments:
            chosen_arm {int} -- chosen article index relative to the pool
            reward {int} -- binary, user click is 1, not click is 0
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
        """

        a = pool_idx[chosen_arm]
        self.counts[a] += 1
        n = self.counts[a]
        self.values[a] = ((n-1)/float(n))*self.values[a] + (1/float(n))*reward
        self.alphas[a] += reward
        self.betas[a] += (1-reward)


class LinUCB():

    def __init__(self, n_arms, alpha):

        # size for A, b matrices is 6*2=12
        d = 12
        self.A = np.array([np.identity(d)] * n_arms)
        self.b = np.zeros((n_arms, d, 1))
        self.alpha = alpha
        self.algo = 'LinUCB (α=' + str(alpha) + ')'

    def choose_arm(self, n_trails, user_feat, pool_idx, art_feat):
        """
        
        Return chosen arm relative to the pool
        
        Arguments:
            n_trails {int} -- number of trails
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
    
        """

        A = self.A[pool_idx]
        b = self.b[pool_idx]
        user = np.array([user_feat] * len(pool_idx))
        art_feat = np.array(art_feat)

        A = np.linalg.inv(A)
        x = np.hstack((user, art_feat[pool_idx]))

        x = x.reshape((len(pool_idx), 12, 1))

        theta = A @ b
        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(
            np.transpose(x, (0, 2, 1)) @ A @ x
            )
        return np.argmax(p)

    def update(self, chosen_arm, reward, user_feat, pool_idx, art_feat):
        """
        
        Update parameters of the algo
        
        Arguments:
            chosen_arm {int} -- chosen article index relative to the pool
            reward {int} -- binary, user click is 1, not click is 0
            user_feat {np.array} -- user feature arrary (1,6)
            pool_idx {np.array} -- indexes for article pool
        """

        a = pool_idx[chosen_arm]
        x = np.hstack((user_feat, art_feat[a]))
        x = x.reshape((12, 1))

        self.A[a] = self.A[a] +  x @ np.transpose(x)
        self.b[a] += reward * x













