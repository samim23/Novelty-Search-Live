# Novelty Search Live
Novelty Search Live is a experimental open-source tool that helps musicians find musical inspiration in Ableton Live. It uses a evolutionary algorithm to continuously evolve new audio-clip and synth-parameter combinations, guided exclusively by Novelty Search. Finally, it takes the countless new musical ideas it has discovered, and generates a map with t-SNE which is interactively browsable. Read more: https://medium.com/@samim/musical-novelty-search-2177c2a249cc

## Usage
After everything is setup and Ableton Live is open, run **python noveltysearchlive.py** from your console to start.
You can tweak the following parameters via the command line:        

--name (Song Name)           
--waittime        
--iterationtime        
--populationsize       
--crossoverrate       
--mutationrate       
--tournamentsize        

And tweak many more settings inside the "noveltysearchlive.py" config section.

## Requirements
- ableton: https://www.ableton.com/
- python: https://www.python.org/
- pylive: https://github.com/ideoforms/pylive
- argparse: https://docs.python.org/3/library/argparse.html
- numpy: https://www.scipy.org/scipylib/download.html
- deap: https://github.com/DEAP/deap
- sklearn: http://scikit-learn.org/stable/install.html
- matplotlib: https://matplotlib.org/faq/installing_faq.html
- annoy: https://github.com/spotify/annoy
