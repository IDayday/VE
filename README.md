This is the open source code of ICML2025 submission paper: Improving Subgoal Planning Policy with Self-Supervised Learning.

## AntMaze

To train SPS on the U-shaped ant maze environment, please run:
```
python train_ant.py --env_name AntU
```

Use this table to run SPS on other ant maze navigation tasks:

| Environment                | --env_name |  
| -------------------------- |:----------:| 
| U-shaped ant maze (default)| AntU       | 
| S-shaped ant maze          | AntFb      | 
| $\Pi$-shaped ant maze      | AntMaze    |
| W-shaped ant maze          | AntFg      |



