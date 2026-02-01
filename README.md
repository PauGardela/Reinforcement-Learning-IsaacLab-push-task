# Reinforcement-Learning-IsaacLab-push-task
# Reinforcement Learning Push Task – IsaacLab (MIRS Master)

This repository contains a **Reinforcement Learning push task** implemented in **IsaacLab**, developed for the **MIRS Master** program.

##IMPORTATN CODES ARE:
\push\push_env_cfg.py
\push\mdp\rewards.py
\push\mdp\observations.py
\push\ur3_config\franka\agents\rsl_rl_ppo_cfg.py

---

## Requirements

You must have the following installed:

- **Isaac Sim 5.1**
- **IsaacLab v2.3.X**

### Recommended Folder Structure

It is recommended to place both **Isaac Sim** and **IsaacLab** inside a single folder called `Isaac`, located in your **Documents** directory:

## Installation

1. Download or clone this GitHub repository.

2. Copy the `push` folder from this repository.

3. Paste the `push` folder into the following directory:
Isaac\IsaacLab-main\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation

---

## Logs Directory Setup

Maybe you need to create (if it does not already exist) the following directory for logs:

Isaac\IsaacLab-main\logs\rsl_rl\push_ur3e

> ⚠️ The folder `push_ur3e` may need to be created manually by the user.


And copy there the 2026-01-29_07-44-49_Fase2_Empuje_V1 folder

---

## Running the Task

1. Open a terminal inside the **IsaacLab-main** folder.

2. Run the following command:

```bash
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Push-UR3e-v0 --num_envs 40
