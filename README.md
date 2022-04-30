# Readme

Codes and data for paper "Environment Design for Biased Decision Makers".



## Simulation

In `simulation` folder, `Experiment.py` is main function for simulation test. Use following command to run experiments. `MDPenvironment.py`, `Agent.py`, `Principal.py` , are for environments, agent models and principal models. Simulation results is stored in `simulation/results` folder.

```python
python Experiment.py baseline # baseline test
python Experiment.py reward_modify # reward function modification test
python Experiment.py action_nudge # action nudge test
```



## Human subject experiment

We recruit 300 individual workers from Amazon Mechanical Turk. Experiment data is stored in `human-subject-experiment/mturk-experiment-data.csv`:

-    `uid` represents anonymous workers, from 0 to 299
-    `treatment` treatment of the worker, from 0-2 (0-baseline, 1-reward function modification, 2-action nudge)
-   `test_idx` test order of games, 6 integers from 0 to 5 
-   `data` actions (worker moves) in each game, first is in test game (not included in `test_idx`)
-   `scores` scores in each game, first is in test game (not included in `test_idx`)

Experiment setting is stored in `human-subject-experiment/experiment-setting` in forms of `.json`.
