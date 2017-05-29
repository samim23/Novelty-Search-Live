############ Imports ############
import live
import time
import threading
import random
import math
import datetime
import argparse
from time import time
import numpy as np
import pickle
from deap import base, creator, tools, algorithms
from sklearn.neighbors import NearestNeighbors
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from annoy import AnnoyIndex


############ Setup ############
config = {
            "songname": "save/mysong",
            "waittime":2,
            "iteration_time": 50,
            "population_size": 50,
            "CXPB": 0.4,
            "MUTPB": 0.4,
            "tournament_size":4,
            "novelity_search_threshold": 10,
            "novelity_log_add_amount": 3,
            "novelity_log_maxlen": 20000,
            "backlog_maxsize": 4000,
            "annoy_amount": 2000,
            "annoy_k": 20,
            "annoy_tree": 20,
            "backlogSave": True,
            "backlogSaveInterval":50,
            "tsne_display":True,
            "tsne_trigger_threshold":5000,
            "tsne_trigger":5000
         }

# setup command line argument parsing
parser = argparse.ArgumentParser(description='Ableton Live Novelty Search Engine')
parser.add_argument('-n', '--name', help='Song Name (default '+str(config["songname"])+')', required=False)
parser.add_argument('-w', '--waittime', help='Wait Time (default '+str(config["waittime"])+')', required=False, type=float)
parser.add_argument('-i', '--iterationtime', help='Iteration Time (default '+str(config["iteration_time"])+')', required=False, type=int)
parser.add_argument('-p', '--populationsize', help='Population Size (default '+str(config["population_size"])+')', required=False, type=int)
parser.add_argument('-c', '--crossoverrate', help='Crosover Rate (default '+str(config["CXPB"])+')', required=False, type=float)
parser.add_argument('-m', '--mutationrate', help='Mutation Rate (default '+str(config["MUTPB"])+')', required=False, type=float)
parser.add_argument('-t', '--tournamentsize', help='Tournament Size (default '+str(config["tournament_size"])+')', required=False, type=int)
args = parser.parse_args()

if args.name is not None: config["songname"] = args.name
if args.waittime is not None: config["waittime"] = args.waittime
if args.iterationtime is not None: config["iteration_time"] = args.iterationtime
if args.populationsize is not None: config["population_size"] = args.populationsize
if args.crossoverrate is not None: config["CXPB"] = args.crossoverrate
if args.mutationrate is not None: config["MUTPB"] = args.mutationrate
if args.tournamentsize is not None: config["tournament_size"] = args.tournamentsize

print "Ableton Live Novelty Search - Song Name: " + config["songname"]

# setup pylive - load/save set on disk, to avoid reload
set = live.Set()
try:
    set.load(config["songname"])
except:
    set.scan(scan_devices = True,scan_clip_names = True,)
    set.save(config["songname"])

############ PyLive Functions ############

def playLiveClip(track,clip,pitch,time,status):
    try:
        clip = set.tracks[track].active_clips[clip]
        clip.set_pitch(pitch)
        clip.play()
        #clip.stop()
        print "playLiveClip: " + str(clip) + " pitch: " + str(pitch) + " time: " + str(time) + " status: " + str(status)
    except:
        print "******* playLiveClip ERROR! track: " + str(track) + " clip: " + str(clip)


# Parse Data and play in Ableton Live, customise
def playTrack(individual):
    paramTotal = 0
    for idx, track in enumerate(set.tracks):
        # set clip parameters
        clip = int(individual[paramTotal])
        pitch = int(individual[paramTotal+1])
        time = int(individual[paramTotal+2])
        status = int(individual[paramTotal+3])
        paramTotal += 4

        # set device parameters
        for idx2, device in enumerate(track.devices):
            for idx3, parameter in enumerate(device.parameters):
                parameter.value = individual[paramTotal]
                paramTotal += 1

        # play clip
        playLiveClip(idx,clip,pitch,time,status)

def backlogSave():
    global generation_backlog
    print "******* SAVE BACKLOG TO FILE *******"
    filename = config["songname"] + "_backlog"
    with open(filename,'w') as f:
        pickle.dump(generation_backlog,f)

def backlogLoad():
    global generation_backlog
    print "******* LOAD BACKLOG FROM FILE *******"
    filename = config["songname"] + "_backlog"

    try:
        generation_backlog = pickle.load(open(filename, 'rb'))
    except (OSError, IOError) as e:
        generation_backlog = []
        print "File not found, skip"

############ Evolution Functions ############

def evaluate(individual):
    global generation_backlog
    global annoy
    global annoy_pop
    global annoy_train
    global config
    global test_db
    score = 0
    score2 = 0

    annoy_amount = config["annoy_amount"] #1000
    annoy_k = config["annoy_k"] #4  # A larger value will give more accurate results, but will take longer time to return.

    # get novelity avarage score against novelity backlog
    if annoy.get_n_items() > 0:
        items,dist = annoy.get_nns_by_vector(individual, annoy_amount,search_k=annoy_k, include_distances=True)
        score += np.average(dist)
    # get novelity avarage score against current population
    if annoy_pop.get_n_items() > 0:
        items2,dist2 = annoy_pop.get_nns_by_vector(individual, annoy_amount,search_k=annoy_k, include_distances=True)
        score += np.average(dist2)
    # get novelity avarage score against novelity backlog
    if annoy_train.get_n_items() > 0:
        items3,dist3 = annoy_train.get_nns_by_vector(individual, annoy_amount,search_k=annoy_k, include_distances=True)
        score -= np.average(dist3)# * (len(test_db)) # training set novelity. set -= for fake multi objective opti, towards train set.
        if score <= 0: score = 0
    return score,
    #return score, score2,


def mutate(individual):
    pitches = [-2,0,2,4,6]
    times = [1,2,4,8,16]
    multi = 0
    pitches_device = [-2,0,2,4,6,8]

    for idx, track in enumerate(set.tracks):
        tracklen = len(track.active_clips)
        if tracklen == 0: print "empty track!"; continue;
        individual[multi + 0] = np.random.randint(0, high=tracklen, size=1)[0] #random.randint(0,tracklen-1)
        individual[multi + 1] = pitches[np.random.randint(0, high=len(pitches), size=1)[0]]  #random.choice(pitches)
        individual[multi + 2] = 2 #random.choice(times)
        individual[multi + 3] = 0 # Status: 0=play,1=stop
        multi += 4

        for devices in track.devices:
            for parameter in devices.parameters:
                if parameter.name.endswith("On"): # is integer
                    individual[multi] = np.random.randint(parameter.minimum,high=parameter.maximum, size=1)[0]
                else: # is float
                    individual[multi] = np.random.uniform(parameter.minimum,parameter.maximum)

                # custom overwrites, add your own here
                if 'Device On' == parameter.name: individual[multi] = 1
                if 'On/Off' in parameter.name: individual[multi] = 1
                if 'On' in parameter.name: individual[multi] = 1
                if 'Volume' == parameter.name: individual[multi] = 0.7
                if 'Volume' in parameter.name: individual[multi] = 127
                if 'Transpose' in parameter.name: individual[multi] = random.choice(pitches_device)
                if 'A Coarse' in parameter.name: individual[multi] = random.randint(0,5)
                if 'B Coarse' in parameter.name: individual[multi] = random.randint(0,5)
                if 'C Coarse' in parameter.name: individual[multi] = random.randint(0,5)
                if 'D Coarse' in parameter.name: individual[multi] = random.randint(0,5)
                if 'Tune' in parameter.name: individual[multi] = random.choice(pitches_device) #random.randint(0,5)

                multi += 1

    return individual,


############ Main Generation Loop ############

# Setup parameters for individual
IND_SIZE = 0
for idx, track in enumerate(set.tracks):
    IND_SIZE += 4
    for device in track.devices:
        IND_SIZE += len(device.parameters)
print "ParameterAmount: " + str(IND_SIZE)

# Setup evolution
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))

creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
#toolbox.register("attr_int", random.randint, 0,1)
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=config["tournament_size"]) # 4

generation_backlog = []
test_db = []
gen_counter = 0

# Setup kmeans
annoy = AnnoyIndex(IND_SIZE)
annoy_pop = AnnoyIndex(IND_SIZE)
annoy_train = AnnoyIndex(IND_SIZE)


def evolve():
    global annoy_pop
    global annoy
    global gen_counter
    global generation_backlog
    global toolbox
    global IND_SIZE
    global config
    random.seed(64)

    # Mutation settings
    iteration_time = config["iteration_time"] #60 # evolution step amount, until play output
    population_size = config["population_size"] #50 150
    CXPB = config["CXPB"] #0.4 # crossover rate
    MUTPB = config["MUTPB"] #0.4 # mutation rate
    novelity_log_maxlen = config["novelity_log_maxlen"] #20000 # max size of backlog
    backlog_maxsize = config["backlog_maxsize"] #3000  # after this threshold, sample subset only - to keep perfomace of kmeans stable
    novelity_search_threshold = config["novelity_search_threshold"] #6 # add novelity randomly during evolution (0>100 chance). higher = more novelity
    novelity_log_add_amount = config["novelity_log_add_amount"] # 3 # add novelity at end of evolution (amount)

    # Stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    verbose = True

    # Create/Reset Population
    population = toolbox.population(n=population_size)

    # Mutate Inital Population within set constrains
    for p in population: mutate(p)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Add to Logbook
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose: print logbook.stream
    gen_counter += iteration_time

    # Evolve for x iterations
    for g in range(iteration_time):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

       # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # rebuild novelity index across this generation
        annoy_pop = AnnoyIndex(IND_SIZE, metric='euclidean')
        for p in offspring:
            annoy_pop.add_item(annoy_pop.get_n_items(), p)
        annoy_pop.build(config["annoy_tree"]) # 10 trees

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        population[:] = offspring

        # Select next generation of population, alternative approaches
        #population[:] = toolbox.select(offspring, population_size)
        #population[:] = toolbox.select(population + offspring, population_size)
        record = stats.compile(population) if stats else {}
        logbook.record(gen=g, nevals=len(invalid_ind), **record)
        if verbose: print logbook.stream

        # Add generation top pick random to backlog - Expensive, careful with overuse.
        chance = np.random.randint(0,100)
        if chance < novelity_search_threshold:
            pop_sorted = sorted(population, key=lambda ind: ind.fitness.values, reverse=True)
            generation_backlog.append(pop_sorted[0])
            print "Novelty Add Random. Score: " + str(p.fitness) + " backlog size: " + str(len(generation_backlog))
            # Add to backlog for Kmeans Scan. If over limit, use random sample
            generation_backlog_temp = generation_backlog
            if len(generation_backlog) > backlog_maxsize:
                generation_backlog_temp = random.sample(generation_backlog, backlog_maxsize)
            # Rebuild Backlog Novelity Index
            annoy = AnnoyIndex(IND_SIZE, metric='euclidean')
            for g in generation_backlog_temp:
                annoy.add_item(annoy.get_n_items(), g)
            annoy.build(config["annoy_tree"]) # 10 trees

    # ______________________________________________________

    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in population]
    length = len(population)
    mean = sum(fits) / length
    print(" Avg:" + str(mean) + "  Min:" + str(min(fits)) + "  Max:" + str(max(fits)) )

    # Check if backlog is in limit, otherwise remove oldest result
    if len(generation_backlog) > novelity_log_maxlen:
        generation_backlog.pop(0)

    # Sort population
    pop_sorted = sorted(population, key=lambda ind: ind.fitness.values, reverse=True)

    # Add Top Novel Individuals to Backlog
    i = 0
    for p in pop_sorted:
        #print p.fitness
        if i < novelity_log_add_amount:
            generation_backlog.append(p)
            print "Novelty Found. Score: " + str(p.fitness) + " backlog size: " + str(len(generation_backlog))
        i += 1

    # Add to backlog for Kmeans Scan. If over limit, use random sample
    generation_backlog_temp = generation_backlog
    if len(generation_backlog) > backlog_maxsize:
        generation_backlog_temp = random.sample(generation_backlog, backlog_maxsize)
    # Rebuild Backlog Novelity Index
    annoy = AnnoyIndex(IND_SIZE, metric='euclidean')
    for g in generation_backlog_temp:
        annoy.add_item(annoy.get_n_items(), g)
    annoy.build(config["annoy_tree"]) # 10 trees

    return pop_sorted


############ Tools ############

def tsne(data):
    fig = plt.figure(dpi=165,frameon=True,tight_layout=True)
    finalplot = []

    def plot_embedding(X):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            #plt.text(X[i, 0], X[i, 1], str(digits.target[i]), color=plt.cm.Set1(y[i] / 10.),fontdict={'weight': 'bold', 'size': 9})
            #plt.text(X[i, 0], X[i, 1], data[i], color="red",fontdict={'weight': 'bold', 'size': 9})
            ax.plot(X[i, 0], X[i, 1], '.', picker=2, markersize=4)  # 3 points tolerance
            d_data =  i
            d_data = data[i]
            d_coordinates = [X[i, 0], X[i, 1]]
            finalplot.append([d_coordinates,d_data])

        plt.xticks([]), plt.yticks([])

    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(data)
    plot_embedding(X_tsne)

    def onpick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        points = tuple(zip(xdata[ind], ydata[ind]))
        print('onpick points:', points)

        for i, item in enumerate(finalplot):
            a1 = item[0]
            a2 = [ xdata[ind[0]], ydata[ind[0]] ]
            if a1 == a2:
                individual = item[1] # HACK TO GET ORIGINAL DATA
                print individual
                playTrack(individual)
                break

    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    #plt.savefig('savefig.png')


# Process Control Data From Ableton Live
def AddToTrain(individual):
    global annoy_train
    global test_db
    global IND_SIZE
    global config

    max_memory = 5
    if set.get_master_volume() == 1:
        print set.get_master_volume()
        set.set_master_volume(0.85)

        test_db.append(individual)
        print "SAVING TO TRAINING SET. TestDB Size: " + str(len(test_db))

        annoy_train = AnnoyIndex(IND_SIZE)
        annoy_train.add_item(annoy_train.get_n_items(), individual)
        annoy_train.build(config["annoy_tree"]) # 10 trees

        if len(test_db) > max_memory:
            test_db.pop(0)
            print "delete old memory entry"

    if set.get_master_volume() == 0:
        test_db = []
        # gen_record = []
        annoy_train = AnnoyIndex(IND_SIZE)
        annoy_train.build(config["annoy_tree"]) # 10 trees
        print "clean set"
        set.set_master_volume(0.85)


############ App Main Loop ############

def main():
    global generation_backlog
    global gen_counter
    global config

    show_tsne = config["tsne_display"] # True
    tsne_trigger_threshold = config["tsne_trigger_threshold"] #1000
    tsne_trigger = config["tsne_trigger"] #1000
    waittime = config["waittime"]
    individual_playing = 0
    set.play(reset = "true") # start live on generation start
    playclip = False
    playclipTrigger = 0
    backlogSaveInterval = config["backlogSaveInterval"] #10
    backlogSaveCounter = 0

    backlogLoad()

    while True:
        print "Evolve Population Generations: " + str(gen_counter) + " Backlog Size: " + str(len(generation_backlog))
        currenttime = time()
        final_population = evolve()
        individual = final_population[0]
        # print( "Best Individual", individual )

        if len(generation_backlog) >= playclipTrigger:
            playclip = True

        if playclip:
            # save backlog
            backlogSaveCounter += 1
            if backlogSaveCounter >= backlogSaveInterval and config["backlogSave"] == True:
                backlogSave()
                backlogSaveCounter = 0

            # wait for next Beat
            currenttime2 = time()
            dif = currenttime2 - currenttime
            print "wait: " + str(waittime - dif) + " waittime: " + str(waittime)
            while dif < (waittime-0.4) and dif > 0:
                currenttime2 = time()
                dif = currenttime2 - currenttime
                #set.wait_for_next_beat()

            set.wait_for_next_beat() # Wait For Next Ableton Beat
            playTrack(individual) # play Beat

            # add to training set
            if individual_playing == 0 or len(test_db) == 0:individual_playing = individual
            #AddToTrain(individual_playing)
            individual_playing = individual

        print "_______________________________________________"

        # TSNE
        if len(generation_backlog) > tsne_trigger and show_tsne:
            d = np.array(generation_backlog)
            tsne(d)
            tsne_trigger += tsne_trigger_threshold




if __name__ == "__main__":
    main()
