<!-- markdownlint-disable MD001 MD010 MD024 MD025 MD049 -->

# Lecture 1

## Architecture

1. Storage:

    - **Data nodes** — data plane, store data  

    - **Masters** — control plane, store metadata, RAFT groups  

    - **Tablet nodes** — data plane, dynamic tables, stateless  

2. Compute:

    - **Exec nodes** — perform computations  

    - **Scheduler/controller agents** — manage computations  

**MapReduce** is used for processing.

Users have API that works on HTTP/RPC.

### Tables

![alt text](notes_images/arc.png)

File - **блобик** with a byte sequence $\to$ why to use files directly? $\to$ use **Static/Dynamic (KV storages) No-SQL tables**.

**Multi-tenancy** - 

- YQL over YT
- ClickHouse over YT
- Spark over YT

![alt text](notes_images/queue.png)

# Lecture 2

![alt text](notes_images/choice_logic.png)
