This is the open source code of ICLR2025 submission paper: Goal-Conditioned Reinforcement Learning with Virtual Experiences.

## AntMaze

To train VE on the U-shaped ant maze environment, please run:
```
python train_ant.py --env_name AntU
```

Use this table to run VET on other ant maze navigation tasks:

| Environment                | --env_name |  
| -------------------------- |:----------:| 
| U-shaped ant maze (default)| AntU       | 
| S-shaped ant maze          | AntFb      | 
| $\Pi$-shaped ant maze      | AntMaze    |
| W-shaped ant maze          | AntFg      |



