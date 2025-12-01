# MC Simulation of an Ice Hockey Game
# MC Class
from params import *
import numpy as np

class Hockey():
    def __init__(self, Length, Width, num_players):
        np.random.seed(10)
        self.Length = Length
        self.Width = Width
        self.red_team = num_players // 2
        self.blue_team = num_players // 2
        self.plot_dir = 'plots' # for animation
        assert self.red_team + self.blue_team == num_players 
        assert num_players % 2 == 0, ("Total Number of Players Must Be Even")
        assert num_players >= 4, ("Add More Players")
        assert num_players == 8, ("Only 4 players per team supported currently")
            
        self.blue_score = 0
        self.red_score = 0
        self.score = [self.red_score,self.blue_score]
        self.has_ball = 1.
        self.no_ball = 0.
        
        self.t = 0. # time (s) initally set here, updated throughout sim
        self.total_time = 3600 # seconds
        self.t_list = [0.0]
        self.action_space = ["carry", "pass", "shoot"]
        self.speed = 2 # m/s
        self.max_dist = np.sqrt(self.Length**2 + self.Width**2)
    
    def set_geometry(self):
        # set the goal as being 10% of the width
        self.bot_goal = .4 * self.Width
        self.top_goal = .6 * self.Width
        x0,y0 = self.Length/2, self.Width/2
        self.starting_position = np.array([x0, y0])
        #vx0, vy0 = 0,0
        self.ball = np.array([x0,y0])
        print(f"Number of Red Players:  {self.red_team}")
        print(f"Number of Blue Players: {self.blue_team}")
        rn = np.random.rand()
        if rn > .5: 
            print("Blue Team Starts with the Ball")
            self.restart_team = "blue"
        else:
            print("Red Team Starts with the Ball")
            self.restart_team = "red"
    
    def create_players(self):
        # define the players
        # each player has a position (x,y) and a velocity (vx,vy)
        self.ball[0], self.ball[1] = self.Length / 2, self.Width / 2
        if self.restart_team.lower() == "blue":
            self.b1 = np.array([self.starting_position[0], self.starting_position[1], self.has_ball])
            self.b2 = np.array([1.5*self.starting_position[0], 1.5*self.starting_position[1], self.no_ball])
            self.b3 = np.array([1.5*self.starting_position[0], self.starting_position[1], self.no_ball])
            self.b4 = np.array([1.5*self.starting_position[0], .5*self.starting_position[1], self.no_ball])
            self.r1 = np.array([.5*self.starting_position[0], self.Width, self.no_ball])
            self.r2 = np.array([.5*self.starting_position[0], 2/3*self.Width, self.no_ball])
            self.r3 = np.array([.5*self.starting_position[0], 1/3*self.Width, self.no_ball])
            self.r4 = np.array([.5*self.starting_position[0], 0, self.no_ball])

        else: 
            self.r1 = np.array([self.starting_position[0], self.starting_position[1], self.has_ball])
            self.r2 = np.array([.5*self.starting_position[0], 1.5*self.starting_position[1], self.no_ball])
            self.r3 = np.array([.5*self.starting_position[0], self.starting_position[1], self.no_ball])
            self.r4 = np.array([.5*self.starting_position[0], .5*self.starting_position[1], self.no_ball])
            self.b1 = np.array([1.5 * self.starting_position[0], self.Width, self.no_ball])
            self.b2 = np.array([1.5*self.starting_position[0], 2/3*self.Width, self.no_ball])
            self.b3 = np.array([1.5*self.starting_position[0], 1/3*self.Width, self.no_ball])
            self.b4 = np.array([1.5*self.starting_position[0], 0, self.no_ball])
        
        self.player_list = [self.b1,self.b2,self.b3,self.b4,
                                self.r1,self.r2,self.r3,self.r4]
    
    @staticmethod
    def Distance(x1,y1,x2,y2): 
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    @staticmethod
    def Speed(distance,time): 
        return distance / time
    
    @staticmethod
    def sample_gamma(Lambda):
        xi1, xi2 = np.random.rand(), np.random.rand()
        return -np.log(xi1 * xi2) / Lambda

    def sample_time(self, Lambda = .5):
        # sample time from an exponential distribution
        # fT(t) = Lambda * e ^ (-Lambda * t), t>= 0
        # xi = 1 - e^(lambda * t), solve for t
        xi = np.random.rand()
        t = -np.log(1 - xi) / Lambda
        self.t += t
        self.t_list.append(self.t.round(5))
        return t
    
    def sample_planck(self,scale = 10):
        rn = np.random.rand()
        rhs = rn * np.pi**4 / 90
        l = 1
        ll = 1
        reject = True
        while reject:
            while l < 100:
                lhs = 0
                for i in range(1,l): lhs += i**(-4)
                if lhs >= rhs:
                    ll = l
                    break
                l += 1
            rn1, rn2, rn3, rn4 = [np.random.rand() for _ in range(4)]
            dist = -1 / ll * (np.log(rn1 * rn2 * rn3 * rn4) * scale)
            if dist < self.max_dist: reject = False
        
        return dist
    
    def sample_shoot(self,d_goal, Lambda):
        # sample the shooting action
        # performed first 
        # returns "shoot" or None
        dshoot = self.shoot_prob(self.max_dist,Lambda)
        #print(dshoot,d_goal)
        return 'shoot' if dshoot >= d_goal else None
    
    @staticmethod
    def shoot_prob(max_dist,Lambda):
        return (-1 / Lambda) * np.log(1 - np.random.rand() * 
                                      (1.0 - np.exp(-Lambda * max_dist)))
                
    def update_score(self):
        self.score = [self.red_score,self.blue_score]
    
    @staticmethod
    def sample_gaussian(mu,sigma): 
        # return one sample in a gaussian centered around a mean distance
        # and given std. dev
        return np.random.normal(mu, sigma, 1)[0]
        
    def show_position(self,shoot_flag, action, possesion):
        # Parameters for easy adjustment
        legend_pad_x = self.Length * 0.25
        pad_y = 1
    
        xmin, xmax = -1, self.Length + legend_pad_x
        ymin, ymax = -1, self.Width + pad_y
    
        plt.figure(figsize=(8, 6))
        plt.plot([0, 0], [0, 0.4*self.Width], color='black')
        plt.plot([0, 0], [0.6*self.Width, self.Width], color='black')
        plt.plot([self.Length, self.Length], [0, 0.4*self.Width], color='black')
        plt.plot([self.Length, self.Length], [0.6*self.Width, self.Width], color='black')
        plt.plot([0, self.Length], [0, 0], color='black')
        plt.plot([0, self.Length], [self.Width, self.Width], color='black')
        plt.plot([self.Length/2, self.Length/2], [0,self.Width], color = 'black')

        # red goalie box
        plt.plot([0,self.Length / 5], [self.Width/4, self.Width/4], color = 'black')
        plt.plot([0,self.Length / 5], [3 * self.Width/4, 3 * self.Width/4], color = 'black')
        plt.plot([self.Length / 5, self.Length/5], [self.Width / 4, 3 * self.Width / 4], color = 'black')

        # blue goalie box
        plt.plot([.8 * self.Length, self.Length], [self.Width/4, self.Width/4], color = 'black')
        plt.plot([.8 * self.Length, self.Length], [3 * self.Width/4, 3 * self.Width/4], color = 'black')
        plt.plot([.8 * self.Length, .8 * self.Length], [self.Width / 4, 3 * self.Width / 4], color = 'black')

    
        # Draw red outline if needed
        if shoot_flag:
            # Draw rectangle using plt.plot for all four sides
            plt.plot([xmin, xmax], [ymax, ymax], color='goldenrod', linewidth=8)  
            plt.plot([xmin, xmax], [ymin, ymin], color='goldenrod', linewidth=8)  
            plt.plot([xmin, xmin], [ymin, ymax], color='goldenrod', linewidth=8)  
            plt.plot([xmax, xmax], [ymin, ymax], color='goldenrod', linewidth=8) 
    
        plt.title(f"Red {self.red_score} - {self.blue_score} Blue. {possesion.upper()} Team Has The Ball")
        minutes = int(self.t / 60)
        seconds = np.round(self.t - 60 * minutes, 2)
        seconds = f"{seconds:05.2f}"
        plt.suptitle(f"Game Time: {minutes}:{seconds}. Action = {action.upper()}") 
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
    
        # Ball plotting
        plt.plot(self.ball[0], self.ball[1], marker='o', markersize=8, color='green', label='Ball')
        blue_players = [
            (self.b1, 'x', 16, 'blue', 'B1'),
            (self.b2, 'x', 16, 'royalblue', 'B2'),
            (self.b3, 'x', 16, 'deepskyblue', 'B3'),
            (self.b4, 'x', 16, 'dodgerblue', 'B4')
        ]
        red_players = [
            (self.r1, 'x', 16, 'red', 'R1'),
            (self.r2, 'x', 16, 'salmon', 'R2'),
            (self.r3, 'x', 16, 'firebrick', 'R3'),
            (self.r4, 'x', 16, 'crimson', 'R4')
        ]
        for player, marker, size, color, label in blue_players + red_players:
            plt.plot(player[0], player[1], marker=marker, markersize=size, color=color, label=label)
    
        plt.legend(loc="center right")
        plt.grid(which='both')
        plt.savefig(f"{self.plot_dir}/position_{round(self.t, 5)}.png")
        plt.close()
    
    def animate_game(self):
        filenames = [f"{self.plot_dir}/position_{self.t_list[i]}.png" 
                     for i in range(len(self.t_list))]
        # Load images
        images = [Image.open(fname) for fname in filenames]        

        # Save as GIF
        images[0].save(
            f"{self.plot_dir}/full_game.gif",
            save_all=True,
            append_images=images[1:],
            duration=200, # duration in ms between frames
            loop=0 # 0 = True       
        )
    
    def dist_to_teammates(self,arrays):
        for i, arr in enumerate(arrays,start=0):
            if arr[2] == self.has_ball: 
                pwb = arr # player with ball (pwb)
                break
            
        distances = np.array([
            self.Distance(pwb[0], pwb[1], p[0], p[1])
            for p in arrays if not np.array_equal(p, pwb) ])
        
        return distances, pwb
    
    def dist_to_opposition(self,pwb,players):
        return np.array([
            self.Distance(pwb[0],pwb[1],p[0],p[1]) for p in players])
    
    def distance_to_goal(self,player,team):
        # find distance from player to center of the goal
        if team.lower() == 'red': return self.Distance(player[0], player[1], 
                                 self.Length, self.Width/2)
        else: return self.Distance(player[0], player[1], 0, self.Width/2)
    
    def check_position(self,arr):
        # confirm that all players are w/in the geometry
        return (False if arr[0] > self.Length or arr[0] < 0 
                or arr[1] > self.Width or arr[1] < 0 else True)
        
    @staticmethod 
    def sample_direction(team):
        # sample angle from -pi/4 to pi/4 towards the goal. 
        angle = np.random.uniform(-np.pi/4,np.pi/4)
        return -angle if team == 'blue' else angle  

    def sample_passing(self,distances):
        # Set scale parameter a (sqrt(kB*T/m)), e.g., a = 1.0
        # return the action and the person passed to
        min_dist = min(distances)
        x = self.sample_gamma(Lambda = .15)
#        x = self.sample_planck(scale=5)
        
        if min_dist > x: return None, None
        
        # normalize probabilities, make cloesest person most probable pass
        dist_arr = (1 / distances) / np.sum(1/distances)
        rn = np.random.rand()
        cdf = np.zeros_like(dist_arr)
        for i in range(dist_arr.size):
            cdf[i] = np.sum(dist_arr[:i+1])
            if rn < cdf[i]: return "pass", distances[i]
    
    def sample_carry(self, team):
        speed = max(.01,self.sample_gaussian(mu=self.speed, 
                                             sigma=self.speed/5)) 
        if speed < 0: raise ValueError("Negative Speed")
        angle = self.sample_direction(team)
        return speed, angle
    
    @staticmethod
    def sample_strip(d_opp,Lambda=.5): 
        # ball is stripped sampling an exponential dist
        return np.exp(-Lambda*d_opp)
    
    @staticmethod
    def two_mins(arr):
        min1 = float('inf')
        min2 = float('inf')
        
        for num in arr:
            if num < min1:
                min2 = min1
                min1 = num
            elif min1 < num < min2: min2 = num
        
        return min1, min2  
    
    def clamp_vector(self,vector):
        # clamp the vector within the bounds
        vector[0] = np.clip(vector[0], 0, self.Length)
        vector[1] = np.clip(vector[1], 0, self.Width)
        
        return vector
    
    def move_attackers_without_ball(self,player_list,goal_mid, t, possesion):
        # keep one player back
        if possesion == 'blue':
            defender_pos = [0,0]
            for v in player_list:
                if v[0] >= defender_pos[0]: defender_pos = v[:2] 

        else: 
            defender_pos = [self.Length,self.Length]
            for v in player_list:
                if v[0] <= defender_pos[0]: defender_pos = v[:2] 

        for v in player_list:
            if v[2] == self.has_ball: continue 
            speed = self.sample_gaussian(self.speed, self.speed/5)
            dist = speed * t

            # if the player is told to stay back
            if v[0] == defender_pos[0] and v[1] == defender_pos[1]: 
                if (v[0] - .8 * self.Length != 0): theta = np.tan((v[1] - self.Width / 2) / (v[0] - .8 * self.Length))
                else: theta = 0
                if possesion == 'blue':
                    if .8 * self.Length > v[0]: v[0] += dist * np.abs(np.cos(theta)) 
                    else: v[0] -= dist * np.abs(np.cos(theta)) 
                    v[0] = max(self.Length/2, v[0])
                else: 
                    if .2 * self.Length > v[0]: v[0] -= dist * np.abs(np.cos(theta)) 
                    else: v[0] += dist * np.abs(np.cos(theta)) 
                    v[0] = min(self.Length/2, v[0])
                if self.Width/2 > v[1]: v[1] += dist * np.abs(np.sin(theta)) 
                else: v[0] -= dist * np.abs(np.sin(theta)) 

                v = self.clamp_vector(v)
                continue

            elif (v[0] - goal_mid[0]) != 0: theta = np.arctan((v[1] - goal_mid[1]) 
                                                            / (v[0] - goal_mid[0])) 
            else: theta = 0
            if goal_mid[0] > v[0]: v[0] += dist * np.abs(np.cos(theta))
            else: v[0] -= dist * np.abs(np.cos(theta))
            if goal_mid[1] > v[1]: v[1] += dist * np.abs(np.sin(theta))
            else: v[1] -= dist * np.abs(np.sin(theta))

            # make sure values are in the domain            
            v = self.clamp_vector(v)
        
        return player_list
    
    def move_defenders(self,player_list, goal_mid, t, possesion, action, overshoot = .25):
        dist_to_ball = []
        for v in player_list: dist_to_ball.append(self.Distance(v[0], v[1], 
                                                self.ball[0], self.ball[1]))
        dist_to_ball = np.array(dist_to_ball)
        min1, min2 = self.two_mins(dist_to_ball)
        
        for i, v in enumerate(player_list):
            # if the players is the closest or 2nd closest to the ball, close
            # in on the ball
            if dist_to_ball[i] == min1 or dist_to_ball[i] == min2:
                #dtb_i = self.Distance(v[0], v[1],self.ball[0], self.ball[1])
                speed = self.sample_gaussian(self.speed, self.speed/4)
                if (v[0] - self.ball[0]) != 0: theta = (np.tan((v[1] - self.ball[1]) 
                                                                / (v[0] - self.ball[0])))
                # if the ball and the player are in the exact same position
                else: 
                    print("Weird thing happening in move defenders")
                    theta = np.random.uniform(0,2*np.pi)
                dist = speed * t
                if self.ball[0] > v[0]: v[0] += dist * np.abs(np.cos(theta))
                else: v[0] -= dist * np.abs(np.cos(theta))
                if self.ball[1] > v[1]:  v[1] += dist * np.abs(np.sin(theta))
                else: v[1] -= dist * np.abs(np.sin(theta))
                v = self.clamp_vector(v)

            # if the defenders are not closest to the ball 
            else: 
                if v[2] == self.has_ball: continue 
                speed = self.sample_gaussian(self.speed, self.speed/5)
                if (v[0] - goal_mid[0]) != 0: theta = np.tan((v[1] - goal_mid[1]) 
                                                                / (v[0] - goal_mid[0])) 
                else: theta = np.random.uniform(0,2*np.pi)
                dist = speed * t

                if goal_mid[0] > v[0]: v[0] += dist * np.abs(np.cos(theta))
                else: v[0] -= dist * np.abs(np.cos(theta))
                if goal_mid[1] > v[1]:  v[1] += dist * np.abs(np.sin(theta))
                else: v[1] -= dist * np.abs(np.sin(theta))

                v = self.clamp_vector(v)
        
        return player_list
        
    def run(self, verbose = False):
        self.set_geometry()
        self.create_players()
        # actions include shoot, pass, strip, carry. 
        # sampling of those actions occur in that order
        action = 'Start'
        actions = 0 # debugging term    
        shoot_flag = False
        if verbose: self.show_position(shoot_flag,action, self.restart_team)
        # don't get stuck in an action loop
        poss_prev = ""
        prev_action = None
        eps = 1e-2
        
        # halftime flags
        half_time = False
        ht_team = 'blue' if self.restart_team == 'red' else 'blue'
        
        while self.t < self.total_time:
            # set up problem for each time step
            actions += 1
            t = self.sample_time()
            #t = self.sample_time(Lambda = .75)
            print("t", self.t.round(3), "s")
            
            # if halftime at 30 minutes
            if self.t >= 1800 and half_time == False:
                print("HALFTIME")
                print(f"Current Score: Red {self.score} Blue")
                # let the other team start with the ball at kickoff
                self.restart_team = ht_team
                self.create_players()
                half_time = True # bool to ensure this happens once

            # search player list to see who has the ball
            count = 0
            for arr in self.player_list:
                if arr[2] == self.has_ball:
                    possesion = "blue"
                    break
                if count >= 3:
                    possesion = "red"
                    break
                count += 1
            if poss_prev != possesion or action == 'strip': 
                print(f"{possesion.upper()} team has the ball")
            poss_prev = possesion
            

            # get distances to teammates, opposition, goal, 
            # and find player with ball
            if possesion == 'blue': 
                poss_list = [self.b1,self.b2,self.b3,self.b4]
                def_list  = [self.r1,self.r2,self.r3,self.r4]
            else:
                poss_list = [self.r1,self.r2,self.r3,self.r4]
                def_list  = [self.b1,self.b2,self.b3,self.b4]
            
            #error checking
            poss_flag = False
            for v in poss_list:
                if v[2] == self.has_ball: poss_flag = True

                    
            if poss_flag == False: raise ValueError("Error in Possesion Block")

            d_teammates, pwb = self.dist_to_teammates(poss_list)
            d_opposition = self.dist_to_opposition(pwb, def_list)
            d_goal = self.distance_to_goal(pwb, possesion)

            
            # sample the action of the player with the ball
            prev_action = action
            action = self.sample_shoot(d_goal,Lambda = .10)
            if action == None and prev_action != 'pass': action, person = self.sample_passing(d_teammates)
            
            if action == 'shoot':
                dshoot = self.shoot_prob(self.max_dist, Lambda = .25)
                #print("IN SHOOT BLOCK")
                #print(dshoot,d_goal)
                if dshoot > d_goal: # score
                #if dshoot < d_goal: # score
                    for v in poss_list: v[2] = 0.0
                    if possesion == 'blue':
                        self.restart_team = 'red'
                        self.blue_score += 1
                    else:
                        self.restart_team = 'blue'
                        self.red_score += 1
                    self.create_players()
                    self.update_score()
                    print(f"{possesion.upper()} Team Scores!")
                    print(f"The Score is now Red {self.score} Blue")
                        
                else: # miss
                    reset_list = []
                    for player in def_list:
                        if possesion == 'red':
                            reset_list.append(self.Distance(player[0], player[1], 
                                             self.Length, self.Width/2))
                        else:
                            reset_list.append(self.Distance(player[0], player[1], 
                                                            0, self.Width/2))
                    min_index = reset_list.index(min(reset_list))

                    for i, arr in enumerate(def_list):
                        if i == min_index: 
                            arr[2] = self.has_ball 
                            self.ball[0], self.ball[1] = arr[0], arr[1]
                    for arr in poss_list: arr[2] = 0.0
                    
                    print(f'{possesion.upper()} Team Misses Their Shot')
                
            elif action == "pass": 
                filtered_vectors = [v for v in poss_list if v[2] != self.has_ball]
                for v in poss_list: v[2] = 0.0
                for i in range(len(filtered_vectors)):
                    if person == d_teammates[i]:
                        self.ball[0] = filtered_vectors[i][0]
                        self.ball[1] = filtered_vectors[i][1]
                        filtered_vectors[i][2] = self.has_ball
                      
            elif action == None:
                # sample distance to nearest opponent, and sample stripping
                closest_opp = np.min(d_opposition)
                closest_opp_ind = np.argmin(d_opposition) 
                pstrip = self.sample_strip(closest_opp, Lambda=.25)
                # ball is stripped
                if np.random.rand() < pstrip and prev_action != 'strip': 
                    for v in poss_list: v[2] = 0.0 # clear index
                    for v in def_list:  v[2] = 0.0
                    for i, v in enumerate(def_list):
                        if i == closest_opp_ind: v[2] = self.has_ball
                    action = 'strip'
                
                # if pass, shoot, and strip fail, do a carry
                else: 
                    speed,angle = self.sample_carry(possesion)   
                    # sample distances based on speed from gaussian
                    dist = speed * t
                    for arr in poss_list:
                        if arr[2] == self.has_ball:
                            if possesion == 'blue':
                                arr[0] += dist * -np.cos(angle)
                                arr[1] += dist *  np.sin(angle)
                            else:
                                arr[0] += dist *  np.cos(angle)
                                arr[1] += dist * -np.sin(angle)
                            self.ball[0] = arr[0]
                            self.ball[1] = arr[1]
                            
                            if not self.check_position(arr):
                                if possesion == 'blue':
                                    arr[0] -= 2 * dist * -np.cos(angle)
                                    arr[1] -= 2 * dist *  np.sin(angle)
                                else:
                                    arr[0] -= 2 * dist *  np.cos(angle)
                                    arr[1] -= 2 * dist * -np.sin(angle)
                                self.ball[0] = arr[0]
                                self.ball[1] = arr[1]
                            
                            break # no need to check the rest of the players
    
                    action = "carry"

            # plot
            shoot_flag = True if action == 'shoot' else False
            self.show_position(shoot_flag, action, possesion)

            # do action for every other player if not a shot or a strip
            # set the vector towards the goal line where the attacking player is headed to
            if action == 'pass' or action == 'carry':
                h1 = np.random.uniform(self.bot_goal, self.top_goal)
                h2 = np.random.uniform(self.bot_goal, self.top_goal)
                # gm: point on the goal line
                gm1 = np.array([self.Length,h1]) if possesion == 'red' else np.array([0,h1])
                gm2 = np.array([self.Length,h2]) if possesion == 'red' else np.array([0,h2])
                
                # attackers w/out ball move towards goal
                poss_list = self.move_attackers_without_ball(poss_list, gm1, t, possesion)
                
                # defenders move towards the ball or towards their goal line
                def_list = self.move_defenders(def_list, gm2, t, possesion, action)

            # printing and shortened runs
            if verbose: print(actions,action.upper())
                
            assert -eps <= self.ball[0] < self.Length + eps
            assert-eps <= self.ball[1] < self.Width + eps

            #if actions > 20: break
            #if self.t > 600: break
            #if self.blue_score != 0 or self.red_score!= 0: break
        
        self.animate_game()

# initialize class
hk = Hockey(Length,Width,num_players)

# run
hk.run(verbose)
print("Game Over!")
print(f"Red {hk.score} Blue")
if hk.blue_score > hk.red_score: print("Blue Team Wins!")
elif hk.blue_score < hk.red_score: print("Red Team Wins!")
else: print("It's a Tie!")
