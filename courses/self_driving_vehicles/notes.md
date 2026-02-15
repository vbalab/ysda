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

If colima is now set:

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
    -v "$PWD/ros2_turtlesim_ws":/home/ubuntu/ws/src \
    tiryoh/ros2-desktop-vnc:jazzy
    ```

2. Then go to [http://127.0.0.1:6080/](http://127.0.0.1:6080/).

3. Open terminator and do:

    ```bash
    ros2 run turtlesim turtlesim_node
    ```


```bash
cd ~/ws
colcon build
source install/local_setup.bash
```

In ROS 2, an **action** definition always has:

- Goal - input sent by the client

- Feedback:  
    Sent zero or many times while it’s running.  
    Used for progress/state updates that the client can observe without waiting for completion.

- Result - sent once at the end when the action finishes.