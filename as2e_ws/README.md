# Neuro Wheel Gazebo Simulation Environment

This project provides the **Neuro Wheel** Gazebo simulation environment, developed on **Ubuntu 22.04** with **ROS 2 Humble**.  
Follow the steps below to run the simulation.

---

## 1. Move the Model Files
Move the **model** folder from the `neuro_wheel` directory to Gazebo’s model directory:

```bash
mv neuro_wheel/model ~/.gazebo/models/
```

---

## 2. Build the Workspace
Navigate to the `as2e_ws` directory and build the workspace using **colcon**:

```bash
cd as2e_ws
colcon build
```

---

## 3. Source the Environment
Once the build is complete, source the environment variables:

```bash
source install/setup.bash
```

---

## 4. Launch the Simulation
To start the simulation, run:

```bash
ros2 launch neuro_wheel as2e_rccar.launch.py
```

---

## 5. Result
After completing these steps, the **Neuro Wheel** Gazebo simulation environment should be running successfully.

---

