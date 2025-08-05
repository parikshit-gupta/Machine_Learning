from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.optimizers import Adam
from collections import deque
import numpy as np

MAX_REPLAY_MEM_SIZE = 10000  # Maximum size of the replay memory
MIN_REPLAY_MEM_SIZE = 1000  # Minimum size of the replay memory before training starts
BATCH_SIZE = 32  # Number of samples from the replay memory to use for training at once
TARGET_UPDATE_AT=100    # number of episodes after which to update target network

class agent():
    def __init__(self, env_size, env_num_actions, stack_frame=2, learning_rate=0.001, gamma=0.99):
        self.env_size = env_size
        self.env_num_actions=env_num_actions
        self.stack_frame = stack_frame  # Number of frames to stack for input to the model
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Create the model
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_mem=deque(maxlen=MAX_REPLAY_MEM_SIZE)
        
        self.target_update_counter=0    #counts number of episodes since last target network update
        self.steps=0
        
        self.model.summary()
        # add tensorboard code
        
    def create_model(self):
        model=Sequential()
        model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', data_format="channels_first", input_shape=(self.stack_frame, self.env_size, self.env_size)))
        ''' """input shape is (height, width, channels) but Keras expects (batch_size, height, width, channels)
        keras handles the batch size automatically, so we don't need to specify it in the input shape
        but we will need to reshape our input data accordingly"""
        '''
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.env_num_actions, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, state, action, reward, next_state, done):
        self.replay_mem.append((state, action, reward, next_state, done))
        
    def train(self, is_terminal_state):
        # this train function will be called at every step to update the main network
        # ocassionally it will also update the target network
        
        if len(self.replay_mem) < MIN_REPLAY_MEM_SIZE:
            return # Not enough samples to train the model
        
        # Sample a batch of experiences from the replay memory
        minibatch_ind = np.random.choice(len(self.replay_mem), BATCH_SIZE, replace=False) #NOTE: np.random.choice is different from random.choice
        
        X=[]
        Y=[]
        for i in minibatch_ind:
            state, action, reward, next_state, done = self.replay_mem[i]
            
            '''
            X must contain the input to the Conv2D input layer, which is infact the state of size (20,20,2)
            Y must be the target Q values Y.shape()=(BATCH_SIZE, 4)
            1 of the 4 values corresponding to the action of the current transition is calculated using the 
            target net, rest are taken as it is from the main network
            
            we basically want to update only that Q-value from the main network which corresponds to the action
            of current transition
            '''
            
            # query main network 
            target_qs=self.get_q_values(state)
            
            # query target network
            if not done:
                target_qs[action]=reward+ self.gamma*np.max(self.target_model.predict(np.array(next_state).reshape(-1, *next_state.shape), verbose=0)[0])
            else:
                target_qs[action]=reward
            
            # build dataset for training
            X.append(state)
            Y.append(target_qs)     
        
        self.model.fit(np.array(X), np.array(Y), batch_size=BATCH_SIZE, verbose=0)
        
        if is_terminal_state:
            self.target_update_counter+=1
        
        if self.target_update_counter>=TARGET_UPDATE_AT:
            self.target_model.set_weights(self.model.get_weights)
            self.target_update_counter=0
    
    # returns the Q-values for the given state by querying the main network, state: (20,20,2)
    def get_q_values(self,state):
        # model.predict() always outputs (batch_size, output_layer_shape)
        return self.model.predict(np.array(state).reshape(-1, *state.shape), verbose=0)[0]
    
    def e_greedy(self, state, epsilon):
        action=0
        if np.random.random() > epsilon:     #with probability 1-e do
            action=np.argmax(self.get_q_values(state))
        else:       #with probability e do
            action=np.random.randint(0,self.env_num_actions)
        return action
        