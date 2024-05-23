# Gymnasium enabled UR5 arm with Robotiq-85 / 140 gripper in PyBullet simulator
Forked from [ElectronicElephant/pybullet_ur5_robotiq](https://github.com/ElectronicElephant/pybullet_ur5_robotiq)

This repo is under active development. Issues / PRs are welcomed.

![User Control Demo](https://raw.githubusercontent.com/ElectronicElephant/pybullet_ur5_robotiq/main/example.png)

## Highlights

- UR5 arm with end-effector 6D IK (Position [X Y Z] and Orientation [R P Y])
- Enhanced Robotiq-85 / 140 gripper, with precise position control and experimental torque control
- Built-in YCB models loader (and obj models after decomposition)
- Gym-styled API, making it suitable for Reinforcement Learning in the field of push-and-grasp
- A heuristic grasping demo
- An interactive user-control demo

## Prerequisites
- Python 3
- PyBullet

## Run

You can try this repo with the interactive user-control demo.
```[Python]
python train.py
```

###  References
https://github.com/ElectronicElephant/pybullet_ur5_robotiq

https://github.com/matafela/pybullet_grasp_annotator_robotiq_85

https://github.com/zswang666/pybullet-playground

https://github.com/ros-industrial/robotiq

https://github.com/Alchemist77/pybullet-ur5-equipped-with-robotiq-140

I do not claim copyright for any model files under this repo.
