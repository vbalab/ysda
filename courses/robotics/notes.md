<!-- markdownlint-disable MD001 MD010 MD024 MD025 MD049 -->

# Workflow

```bash
ssh balabaevvl@45.93.203.34
ssh shad-gpu

# scp /path/to/local/file username@remote_host:/path/to/remote/destination
scp main.cu shad-gpu:cuda/
scp notes.md shad-gpu:cuda/

# scp username@remote_host:/path/to/remote/file /path/to/local/destination
scp -r shad-gpu:cuda/* .
```

```bash
nvidia-smi                                                              # see GPUs
nvidia-smi -L
```

```bash
tmux
tmux new -s isaac
tmux attach -t isaac
```

[Isaac Sim Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html#isaac-sim-app-install-pip)

```bash
python3.11 -m venv env_isaacsim
source env_isaacsim/bin/activate

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install isaacsim[all,extscache]==5.0.0 --extra-index-url https://pypi.nvidia.com

pip show isaacsim
```

```bash
isaacsim
```

Download [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/download.html#isaac-sim-latest-release) locally on your laptop.

```bash
isaacsim isaacsim.exp.full.streaming --no-window
isaacsim isaacsim.exp.full.streaming --no-window --/app/livestream/publicEndpointAddress=45.93.203.34 --/app/livestream/port=49100
```

```bash
ps -ef | grep isaacsim
kill -KILL ...
```

---

```bash
isaacsim ... 2>&1 | tee isaacsim.log    # server
scp shad-gpu:isaacsim.log .             # laptop
```

---

# **Lecture 1**

## Spaces

- **Configuration Space** — set of all possible configurations of the body/mechanism/robot, most of the times, this set has a nice structure (manifold, or even a Lie group).

- **End-effector** — particular point/link/attachment on the robot, usually the one that performs the task.

- **Task Space** — set of all possible task-related configurations of the end-effector or of positions of relevant robot parts, e.g. position and orientation of scalpel for surgical robot, or grasping configurations of a human hand around the cup, usually a subset of $\mathbb{R}^n$. Sometimes there are more than one configurations of robot that achieve specific end-effector configuration, and sometimes there is none.

- **Work Space** — set of all reachable positions of the end-effector, usually a subset of $\mathbb{R}^n$.

Examples:

- $R^2 \times S^1$ $(x, y; \theta)$
- $R^3 \times S^3$ $(x, y, z; \theta, \phi, \gamma)$ - SE(3)

## 2D Rigid Body

A **rigid body** in the plane is an object whose points keep fixed distances and angles; only position and orientation can change. The set of all such transformations is the **2D Euclidean group** $\mathrm{SE}(2)$. It is built from **translations** and **rotations**.

### Translations

**Translation** by a vector $d = (d_x, d_y) \in \mathbb{R}^2$ moves every point $p$ to $p' = p + d$. The set of all 2D translations is $\mathbb{R}^2$ with addition. Translations preserve distances and orientations (no rotation).

### Rotations

**Rotation** about the origin by angle $\theta$ (counterclockwise) is a linear map given by the matrix
$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}.
$$
These matrices form the **special orthogonal group** $\mathrm{SO}(2)$: $2\times 2$ matrices with $R^\top R = I$ and $\det R = 1$. So $\mathrm{SO}(2)$ is the group of 2D rotations (one DOF: $\theta \in S^1$).

The defining identities imply preservation of the **Euclidean metric**. For any vectors $p, q \in \mathbb{R}^2$, the condition $R^\top R = I$ gives
$$
\langle R p,\, R q \rangle = (R p)^\top (R q) = p^\top R^\top R\, q = p^\top q = \langle p,\, q \rangle.
$$
So the dot product, and hence lengths $\|R p\| = \|p\|$ and angles, via $\alpha = \arccos\left(\frac{\langle p, q \rangle}{\|p\| \|q\|}\right)$, are preserved. The condition $\det R = 1$ restricts to **proper** rotations (orientation preserved); $\det R = -1$ would give a reflection, which also preserves the metric but reverses orientation.

### Homogeneous Coordinates

To **combine** rotation and translation in one operation we use **homogeneous coordinates**. A point $p = (x, y) \in \mathbb{R}^2$ is represented as $\tilde{p} = (x, y, 1)^\top$. A **rigid transformation** (rotation $R \in \mathrm{SO}(2)$ and translation $d \in \mathbb{R}^2$) is encoded as a $3\times 3$ matrix
$$
T = \begin{bmatrix} R & d \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta & d_x \\ \sin\theta & \cos\theta & d_y \\ 0 & 0 & 1 \end{bmatrix}.
$$
Then the transformed point is $\tilde{p}' = T \tilde{p}$: the top two components are $p' = R p + d$. So: first rotate, then translate. Composition of two rigid motions is matrix multiplication: $T_{ac} = T_{ab} T_{bc}$.

### Euclidean

The **2D Euclidean group** $\mathrm{SE}(2)$ is the set of all rigid motions (rotations + translations) with composition as the group operation. As matrices:
$$
\mathrm{SE}(2) = \left\{ T = \begin{bmatrix} R & d \\ 0 & 1 \end{bmatrix} : R \in \mathrm{SO}(2),\; d \in \mathbb{R}^2 \right\}.
$$
It is the **semidirect product** $\mathrm{SO}(2) \ltimes \mathbb{R}^2$. Inverse:
$$
T^{-1} = \begin{bmatrix} R^\top & -R^\top d \\ 0 & 1 \end{bmatrix}
$$
(inverse rotation, then inverse translation in the rotated frame).

## 3D Rigid Body

A **rigid body** in 3D is an object whose points keep fixed distances and angles; only position and orientation change. The configuration is given by $(R, p)$ with $R \in \mathrm{SO}(3)$ and $p \in \mathbb{R}^3$.

**SO(3)** — the **special orthogonal group** — is the set of all $3\times 3$ rotation matrices satisfying $R^\top R = I$ and $\det R = 1$. It is a Lie group of dimension 3. Metric preservation follows: $\langle R u,\, R v \rangle = \langle u,\, v \rangle$ (lengths and angles preserved); $\det R = 1$ restricts to proper rotations (no reflection).

**SE(3)** — the **special Euclidean group** — is the set of all rigid motions, represented as $4\times 4$ matrices
$$
T = \begin{bmatrix} R & p \\ 0 & 1 \end{bmatrix}, \quad R \in \mathrm{SO}(3),\; p \in \mathbb{R}^3.
$$
Action on a point: $p' = R\, p + p_{\text{trans}}$ (rotation then translation; $p_{\text{trans}}$ is the translation block of $T$). Composition is matrix multiplication; the inverse is
$$
T^{-1} = \begin{bmatrix} R^\top & -R^\top p \\ 0 & 1 \end{bmatrix}.
$$

### Euler Angles

Euler angles build a 3D rotation by composing three elementary rotations about body (or fixed) axes. Many conventions exist (ZYX, ZYZ, etc.); a common one in robotics is **ZYX (roll–pitch–yaw)**:
$$
R = R_z(\alpha)\, R_y(\beta)\, R_x(\gamma).
$$
The elementary rotation matrices (with $c = \cos$, $s = \sin$) are
$$
R_x(\gamma) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & c_\gamma & -s_\gamma \\ 0 & s_\gamma & c_\gamma \end{bmatrix}, \quad
R_y(\beta) = \begin{bmatrix} c_\beta & 0 & s_\beta \\ 0 & 1 & 0 \\ -s_\beta & 0 & c_\beta \end{bmatrix}, \quad
R_z(\alpha) = \begin{bmatrix} c_\alpha & -s_\alpha & 0 \\ s_\alpha & c_\alpha & 0 \\ 0 & 0 & 1 \end{bmatrix}.
$$
The full rotation is then $R = R_z(\alpha)\, R_y(\beta)\, R_x(\gamma)$ (applied right-to-left: first $R_x(\gamma)$, then $R_y(\beta)$, then $R_z(\alpha)$).

At $\beta = \pm\pi/2$ the ZYX Euler representation becomes singular (**gimbal lock**).

#### Gimbal Lock

When $\beta = \pm\pi/2$ in the ZYX Euler convention, the axes of the first rotation ($R_z$) and the third ($R_x$, applied in the body after $R_y$) align. The composition then depends only on the sum $\alpha + \gamma$: one degree of freedom is lost, and $R$ is unchanged under $(\alpha,\gamma) \mapsto (\alpha + \delta, \gamma - \delta)$. Explicitly, for $\beta = \pi/2$ we have
$$
R = R_z(\alpha)\, R_y(\pi/2)\, R_x(\gamma) = R_z(\alpha+\gamma)\, R_y(\pi/2),
$$
so the third column of $R$ is $\bigl(\sin(\alpha+\gamma),\, -\cos(\alpha+\gamma),\, 0\bigr)^\top$.

**Demo:** β is fixed at $+\pi/2$ or $-\pi/2$; vary α and γ with the sliders. The **magenta/yellow/cyan** frame is $R = R_z(\alpha)R_y(\beta)R_x(\gamma)$. The title shows $\alpha+\gamma$; different pairs $(\alpha,\gamma)$ with the same sum give the same orientation.

## Quaternions

A **unit quaternion** $q = (q_w, q_x, q_y, q_z)$ represents a rotation when $q_w^2 + q_x^2 + q_y^2 + q_z^2 = 1$. The scalar part is $q_w$; the vector part is $\vec{q} = (q_x, q_y, q_z)$.

**Rotating a vector**: For $v \in \mathbb{R}^3$, the rotated vector is $v' = q \odot v \odot q^{-1}$ (quaternion product with $v$ treated as pure quaternion $(0, v)$). Equivalently one converts $q$ to a $3\times 3$ rotation matrix $R(q)$; the standard formula in terms of $q_w, q_x, q_y, q_z$ yields $R$ so that $v' = R\, v$.

**Properties**: Unit quaternions form a double cover of SO(3): $q$ and $-q$ give the same $R$. There is no gimbal lock. **Slerp** (spherical linear interpolation) gives smooth interpolation between two rotations.

---

**Gimbal**
**Gimbal Lock**

Кватернионы (единичной длины)

---

Прямая кинематика

Revolutionary Joint

произведение A дает end effector position M

Inverse Kinematics

Численные методы

$$||\hat{M} - M||^2_F$$

# **Lecture 2**

анзац - предполагаемое решение

конфигурации + скорости = фазовое пространство

Euler Scheme (using x & t) - двухточечная явная схема

> в симуляторе используй кватернионы

1я таска в дз - можно почти скопировать с 1го семинара (код прям перед гифки авто с дрифтом)
