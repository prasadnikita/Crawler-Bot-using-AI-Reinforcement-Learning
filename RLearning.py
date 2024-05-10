import random
import numpy as np
import math as mth

# The state class
class State:
    def __init__(self, angle1=0, angle2=0):
        self.angle1 = angle1
        self.angle2 = angle2

class ReinforceLearning:

    #
    def __init__(self, unit=5):

        ####################################  Needed: here are the variable to use  ################################################

        # The crawler agent
        self.crawler = 0

        # Number of iterations for learning
        self.steps = 1000

        # learning rate alpha
        self.alpha = 0.2

        # Discounting factor
        self.gamma = 0.95

        # E-greedy probability
        self.epsilon = 0.1

        self.Qvalue = []  # Update Q values here
        self.unit = unit  # 5-degrees
        self.angle1_range = [-35, 55]  # specify the range of "angle1"
        self.angle2_range = [0, 180]  # specify the range of "angle2"
        self.rows = int(1 + (self.angle1_range[1] - self.angle1_range[0]) / unit)  # the number of possible angle 1
        self.cols = int(1 + (self.angle2_range[1] - self.angle2_range[0]) / unit)  # the number of possible angle 2

        ########################################################  End of Needed  ################################################



        self.pi = [] # store policies
        self.actions = [-1, +1] # possible actions for each angle

        # Controlling Process
        self.learned = False



        # Initialize all the Q-values
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.Qvalue.append(row)



        # Initialize all the action combinations
        self.actions = ((-1, -1), (-1, 0), (0, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))


        # Initialize the policy
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(random.randint(0, 8))
            self.pi.append(row)





    # Reset the learner to empty
    def reset(self):
        self.Qvalue = [] # store Q values
        self.R = [] # not working
        self.pi = [] # store policies

        # Initialize all the Q-values
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.Qvalue.append(row)

        # Initiliaize all the Reward
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.R.append(row)

        # Initialize all the action combinations
        self.actions = ((-1, -1), (-1, 0), (0, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))


        # Initialize the policy
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(random.randint(0, 8))
            self.pi.append(row)

        # Controlling Process
        self.learned = False

    # Set the basic info about the robot
    def setBot(self, crawler):
        self.crawler = crawler


    def storeCurrentStatus(self):
        self.old_location = self.crawler.location
        self.old_angle1 = self.crawler.angle1
        self.old_angle2 = self.crawler.angle2
        self.old_contact = self.crawler.contact
        self.old_contact_pt = self.crawler.contact_pt
        self.old_location = self.crawler.location
        self.old_p1 = self.crawler.p1
        self.old_p2 = self.crawler.p2
        self.old_p3 = self.crawler.p3
        self.old_p4 = self.crawler.p4
        self.old_p5 = self.crawler.p5
        self.old_p6 = self.crawler.p6

    def reverseStatus(self):
        self.crawler.location = self.old_location
        self.crawler.angle1 = self.old_angle1
        self.crawler.angle2 = self.old_angle2
        self.crawler.contact = self.old_contact
        self.crawler.contact_pt = self.old_contact_pt
        self.crawler.location = self.old_location
        self.crawler.p1 = self.old_p1
        self.crawler.p2 = self.old_p2
        self.crawler.p3 = self.old_p3
        self.crawler.p4 = self.old_p4
        self.crawler.p5 = self.old_p5
        self.crawler.p6 = self.old_p6



    def updatePolicy(self):
        # After convergence, generate policy y
        for r in range(self.rows):
            for c in range(self.cols):
                max_idx = 0
                max_value = -1000
                for i in range(9):
                    if self.Qvalue[r][9 * c + i] >= max_value:
                        max_value = self.Qvalue[r][9 * c + i]
                        max_idx = i

                # Assign the best action
                self.pi[r][c] = max_idx


    # This function will do additional saving of current states than Q-learning
    def onLearningProxy(self, option):
        self.storeCurrentStatus()
        if option == 0:
            self.onMonteCarlo()
        elif option == 1:
            self.onTDLearning()
        elif option == 2:
            self.onQLearning()
        self.reverseStatus()


        # Turn off learned
        self.learned = True



    # For the play_btn uses: choose an action based on the policy pi
    def onPlay(self, ang1, ang2, mode=1):

        epsilon = self.epsilon

        ang1_cur = ang1
        ang2_cur = ang2

        # get the state index
        r = int((ang1_cur - self.angle1_range[0]) / self.unit)
        c = int((ang2_cur - self.angle2_range[0]) / self.unit)

        # Choose an action and udpate the angles
        idx, angle1_update, angle2_update = self.chooseAction(r=r, c=c)
        ang1_cur += self.unit * angle1_update
        ang2_cur += self.unit * angle2_update

        return ang1_cur, ang2_cur

    # This function is similar to the "runReward()" function but without returning a reward.
    # It only update the robot position with the new input "angle1" and "angle2"
    def setBotAngles(self, ang1, ang2):
        self.crawler.angle1 = ang1
        self.crawler.angle2 = ang2
        self.crawler.posConfig()

    def get_updated_angles(self, idx):
        if idx == 0:
            angle1_update = -1
            angle2_update = -1
        elif idx == 1:
            angle1_update = -1
            angle2_update = 0
        elif idx == 2:
            angle1_update = -1
            angle2_update = 1
        elif idx == 3:
            angle1_update = 0
            angle2_update = -1
        elif idx == 4:
            angle1_update = 0
            angle2_update = 0
        elif idx == 5:
            angle1_update = 0
            angle2_update = 1
        elif idx == 6:
            angle1_update = 1
            angle2_update = -1
        elif idx == 7:
            angle1_update = 1
            angle2_update = 0
        elif idx == 8:
            angle1_update = 1
            angle2_update = 1
        return angle1_update, angle2_update



    # Given the current state, return an action index and angle1_update, angle2_update
    # Return value
    #  - index: any number from 0 to 8, which indicates the next action to take, according to the e-greedy algorithm
    #  - angle1_update: return the angle1 new value according to the action index, one of -1, 0, +1
    #  - angle2_update: the same as angle1


    def get_updated_angles(self, idx):
        if idx == 0:
            angle1_update = -1
            angle2_update = -1
        elif idx == 1:
            angle1_update = -1
            angle2_update = 0
        elif idx == 2:
            angle1_update = -1
            angle2_update = 1
        elif idx == 3:
            angle1_update = -1
            angle2_update = 1
        elif idx == 4:
            angle1_update = -1
            angle2_update = 1
        elif idx == 5:
            angle1_update = -1
            angle2_update = 1
        elif idx == 6:
            angle1_update = -1
            angle2_update = 1
        elif idx == 7:
            angle1_update = -1
            angle2_update = 1
        elif idx == 8:
            angle1_update = -1
            angle2_update = 1
        return angle1_update, angle2_update

    def chooseAction(self, r, c):
        # implementation here

        # below is just an example of randomly generating angle updates
        if self.epsilon >= random.random():
            idx = random.randint(0, 8)
        else:
            temp_max = self.Qvalue[r][c]
            idx = 0
            i = 0
            while i < 9:
                if self.Qvalue[r][c * 9 + i] > temp_max:
                    temp_max = self.Qvalue[r][c * 9 + i]
                    idx = i
                i += 1

        angle1_update, angle2_update = self.get_updated_angles(idx)

        # if out of the range, then just make angle1_update = 0
        if angle1_update * self.unit + self.crawler.angle1 < self.angle1_range[
            0] or angle1_update * self.unit + self.crawler.angle1 > self.angle1_range[1]:
            angle1_update = 0

        # if out of the range, then just make angle2_update = 0
        if angle2_update * self.unit + self.crawler.angle2 < self.angle2_range[
            0] or angle2_update * self.unit + self.crawler.angle2 > self.angle2_range[1]:
            angle2_update = 0

        return idx, angle1_update, angle2_update

    def get_idx(self, a):
        if a[0] == -1 and a[1] == -1:
            idx = 0
        elif a[0] == -1 and a[1] == 0:
            idx = 1
        elif a[0] == -1 and a[1] == 1:
            idx = 2
        elif a[0] == 0 and a[1] == -1:
            idx = 3
        elif a[0] == 0 and a[1] == 0:
            idx = 4
        elif a[0] == 0 and a[1] == 1:
            idx = 5
        elif a[0] == 1 and a[1] == -1:
            idx = 6
        elif a[0] == 1 and a[1] == 0:
            idx = 7
        elif a[0] == 1 and a[1] == 1:
            idx = 8

        return idx

    def get_traj(self, s_new):
        j = 0
        t = []
        while (j <= 30 and s_new[0] >= 0 and s_new[0] <= 18 and s_new[1] >= 0 and s_new[1] <= 36):
            t.append(s_new)
            s = s_new
            a1 = random.randint(-1, 1)
            a2 = random.randint(-1, 1)
            s_new = [s[0] + a1, s[1] + a2]
            self.setBotAngles(s_new[0] * 5 + self.angle1_range[0], s_new[1] * 5 + self.angle2_range[0])
            t.append([a1, a2])
            j += 1
        return t

# Method 1: Monte Carlo algorithm
    def onMonteCarlo(self):
        # You need to implement this function for the project 4 part 1
        trajectories = []
        R = [[0] * 333] * 19
        N = [[0] * 333] * 19
        for i in range(self.steps):
            org_location = self.crawler.location[0]

            r = random.randint(0, 18)
            c = random.randint(0, 36)
            s_new = [r, c]

            t = self.get_traj(s_new)

            terminated_location = self.crawler.location[0]
            trajectories.append(t)
            utility = terminated_location - org_location

            if utility != 0:
                k = 0
                for x in range(0, len(t) // 2, 2):
                    R[t[x][0]][t[x][1] * 9 + self.get_idx(t[x + 1])] += self.gamma * (len(t) // 2 - k) * utility
                    N[t[x][0]][t[x][1] * 9 + self.get_idx(t[x + 1])] += 1
                    k += 1

        for row in range(self.rows):
            for col in range(self.cols):
                for i in range(9):
                    if N[row][col * 9 + i] != 0:
                        self.Qvalue[row][col * 9 + i] = R[row][col * 9 + i] / N[row][col * 9 + i]

    def find_idx(self, a):
        if a[0] == -1 and a[1] == -1:
            idx = 0
        elif a[0] == -1 and a[1] == 0:
            idx = 1
        elif a[0] == -1 and a[1] == 1:
            idx = 2
        elif a[0] == 0 and a[1] == -1:
            idx = 3
        elif a[0] == 0 and a[1] == 0:
            idx = 4
        elif a[0] == 0 and a[1] == 1:
            idx = 5
        elif a[0] == 1 and a[1] == -1:
            idx = 6
        elif a[0] == 1 and a[1] == 0:
            idx = 7
        elif a[0] == 1 and a[1] == 1:
            idx = 8

        return idx

# Method 2: Temporal Difference based on SARSA
    def onTDLearning(self):
        # You don't have to work on it for the moment
        global R
        global N
        # You need to implement this function for the project 4 part 1
        trajectories = []

        for i in range(self.steps):
            t = []
            r = random.randint(0, 18)
            c = random.randint(0, 36)
            new_state = [r, c]

            j = 0
            while (j < 5 and new_state[0] >= 0 and new_state[0] <= 18 and new_state[1] >= 0 and new_state[1] <= 36):
                t.append(new_state)
                self.setBotAngles(new_state[0] * 5 + self.angle1_range[0],
                                  new_state[1] * 5 + self.angle2_range[0])  # for SARSA
                org_location = self.crawler.location[0]  # for SARSA
                s = new_state
                idx, a1, a2 = self.chooseAction(s[0], s[1])
                new_state = [s[0] + a1, s[1] + a2]
                self.setBotAngles(new_state[0] * 5 + self.angle1_range[0], new_state[1] * 5 + self.angle2_range[0])
                terminated_location = self.crawler.location[0]  # for SARSA
                immediate_reward = terminated_location - org_location  # for SARSA
                t.append([a1, a2])
                t.append(immediate_reward)  # for SARSA
                j = j + 1

                self.update_Qvalues(t)

    def update_Qvalues(self, t):
        for x in range(0, len(t) - 3, 3):
            self.Qvalue[t[x][0]][(t[x][1]) * 9 + self.find_idx(t[x + 1])] = self.Qvalue[t[x][0]][
                                                                                (t[x][1]) * 9 + self.find_idx(
                                                                                    t[x + 1])] + self.alpha * (
                                                                                        t[x + 2] + self.gamma *
                                                                                        self.Qvalue[t[x + 3][0]][(t[
                                                                                            x + 3][
                                                                                            1]) * 9 + self.find_idx(
                                                                                            t[x + 4])] -
                                                                                        self.Qvalue[t[x][0]][(t[x][
                                                                                            1]) * 9 + self.find_idx(
                                                                                            t[x + 1])])

# Method 3: Bellman operator based Q-learning
    def onQLearning(self):
        # You don't have to work on it for the moment
        # You need to implement this function for the project 4 part 1

        for i in range(self.steps):
            t = []
            r = random.randint(0, 18)
            c = random.randint(0, 36)
            s_new = [r, c]
            j = 0
            while (j < 5 and s_new[0] >= 0 and s_new[0] <= 18 and s_new[1] >= 0 and s_new[1] <= 36):
                t.append(s_new)
                self.setBotAngles(s_new[0] * 5 + self.angle1_range[0], s_new[1] * 5 + self.angle2_range[0])
                org_location = self.crawler.location[0]  # for SARSA
                s = s_new
                idx, a1, a2 = self.chooseAction(s[0], s[1])
                s_new = [s[0] + a1, s[1] + a2]
                self.setBotAngles(s_new[0] * 5 + self.angle1_range[0], s_new[1] * 5 + self.angle2_range[0])
                terminated_location = self.crawler.location[0]  # for SARSA
                immediate_reward = terminated_location - org_location  # for SARSA
                t.append([a1, a2])
                t.append(immediate_reward)  # for SARSA
                j = j + 1
            self.upgrade_Qval(t, r, c)

    def upgrade_Qval(self, t, r, c):
        for x in range(0, len(t) - 3, 3):
            max = self.Qvalue[t[x + 3][0]][(t[x + 3][1]) * 9]
            for i in range(9):
                if self.Qvalue[r][c * 9 + i] > max:
                    max = self.Qvalue[t[x + 3][0]][(t[x + 3][1]) * 9 + i]
            self.Qvalue[t[x][0]][(t[x][1]) * 9 + self.find_idx(t[x + 1])] = self.Qvalue[t[x][0]][
                                                                                (t[x][1]) * 9 + self.find_idx(
                                                                                    t[x + 1])] + self.alpha * (
                                                                                        t[x + 2] + self.gamma * max -
                                                                                        self.Qvalue[t[x][0]][(t[x][
                                                                                            1]) * 9 + self.find_idx(
                                                                                            t[x + 1])])
