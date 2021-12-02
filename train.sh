#!/bin/bash

python train.py --project 'MazeEnvSample5x5' --env 'MazeEnvSample5x5' --episode 1500 --ngu --horizon 100

python train.py --project 'MazeEnvSample5x5' --env 'MazeEnvSample5x5' --episode 1500 --horizon 100

python train.py --project 'MazeEnvRandom5x5' --env 'MazeEnvRandom5x5' --episode 1500 --ngu --horizon 100

python train.py --project 'MazeEnvRandom5x5' --env 'MazeEnvRandom5x5' --episode 1500  --horizon 100

python train.py --project 'MazeEnvSample10x10' --env 'MazeEnvSample10x10' --episode 6000 --ngu --horizon 400