<!-- markdownlint-disable MD024 MD025 -->

# **Lecture 01 - Area Intro**

...

# **Homework 01**

## Run Docker on Mac

```bash
colima list

colima start
colima status
colima stop

colima start --cpu 4 --memory 8 --disk 100
colima start --edit
```

Check if colima is set:

```bash
docker context ls
docker context show
```

If colima is not set:

```bash
docker context use colima
unset DOCKER_HOST
```

## Setup

1. Do:

    ```bash
    docker run -p 6080:80 --shm-size=512m tiryoh/ros2-desktop-vnc:jazzy

    # or, with mount:
    docker run -p 6080:80 --shm-size=512m \
    -v "$PWD/course/hw01_turtle/ros2_turtlesim_src":/home/ubuntu/ws/src \
    tiryoh/ros2-desktop-vnc:jazzy

    # or start existing container:
    docker start <name>
    ```

2. Then go to [http://127.0.0.1:6080/](http://127.0.0.1:6080/).  
    (If the screen is locked, password is "ubuntu")

3. Open terminator and do:

    ```bash
    ros2 run turtlesim turtlesim_node
    ```

```bash
cd ~/ws
colcon build
source install/local_setup.bash

# or make auto source:
echo "source /home/ubuntu/ws/install/local_setup.bash" >> ~/.bashrc
```

### VSCode

At the workspace root (in `ws`, same level as `src/`), create a file named `.env`:

```env
PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages
```

## Smth

In ROS 2, an **action** definition always has:

- Goal - input sent by the client

- Feedback:  
    Sent zero or many times while it’s running.  
    Used for progress/state updates that the client can observe without waiting for completion.

- Result - sent once at the end when the action finishes.

```bash
# from one terminal:
ros2 run turtlesim turtlesim_node

# from another terminal:
ros2 run turtlesim_contest_evaluation turtlesim_contest_hide_gift_node 5 5 2 4

# from another terminal:
ros2 run turtlesim_contest_submission turtlesim_contest_submission_node

# from another terminal:
ros2 action send_goal /find_hidden_gift turtlesim_contest_interface/action/FindHiddenGift "{search_area: {bottom_left: {x: 0, y: 0}, top_right: {x: 11, y: 11}}}"
```

# **Lecture 2**

Main localization sensors:

- GNSS

- Lidar
Movement correction of lidar (10Hz, 100km/h, `do we take into account not only time but also speed?`)
bad with micro objects - snow

- Inertial Measurment Unit (IMU) - bad at path increments intergration

- Odometry

эффект доплера, радар
у радара есть преимущество (в отл. от лидара) - он видит скорость $\to$ решая систему, можно узнать нашу скорость (эту скорость можно сравнивать с датчиками)

---

Sensor Fusion (основано на баесовской фильтрации - чекнуть, kalman)

EKF-localization

---

$p(\vec{x}(t)|\vec{z}(t))$

$p(\vec{x}(t + \Delta t)|\vec{x}(t))$

модель наблюдений, модель эволюции
