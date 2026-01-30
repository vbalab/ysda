
## Perception

PointNet works on point clouds.

Point Transformer

Lidar Points Filtering


**BEV (Bird’s-Eye View)** - top-down grid over $(x,y)$.

## Semantic occupancy

2d pixel $\to$ 3d **voxel**.

Instead of just: occupied / free, you want: what kind of thing occupies this cell.

“Lift, Splat, Shoot” 2020 $\to$ “Perceive, Predict, and Plan” 2023.

# UnO

1. There were prior works that directly predicted future point clouds from past point clouds
2. There were prior works that predicted BEV representations (often semantic, often supervised)

UNO instead learns a self-supervised continuous 4D occupancy world model, using BEV only as an internal latent space, and derives point clouds (and BEV semantics) from that model.

Then uses 4D occupancy world model to do point cloud forecasting as a downstream task.
