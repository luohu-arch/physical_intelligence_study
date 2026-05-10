# VLA Research Scoring Rubric

Use this rubric when deciding whether to add a new candidate to the local library.

## Score

- 5: Must read. Changes the main VLA/Physical AI learning map or introduces a widely reusable open model/data/training method.
- 4: Strong read. Important architecture, benchmark, data, adaptation, or action representation idea.
- 3: Useful context. Narrow but relevant; add to watchlist, read later.
- 2: Tangential. Keep only if it supports a specific question.
- 1: Skip. Not directly about physical action, robot learning, embodied control, or VLA systems.

## High-Value Signals

- Has arXiv PDF, project page, code, model weights, or dataset.
- Evaluates on robot manipulation benchmarks or real robots.
- Improves OpenVLA, Octo, pi0/FAST, RT-X, LIBERO, CALVIN, SimplerEnv, RoboTwin, ALOHA, DROID, or Open X-Embodiment baselines.
- Addresses one of the core bottlenecks:
  - action representation
  - data scarcity
  - cross-embodiment generalization
  - long-horizon planning
  - closed-loop robustness
  - low-latency deployment
  - tactile/contact-rich manipulation
  - RL/self-improvement

## Red Flags

- Only claims "agentic" or "embodied" without robot action experiments.
- Only a benchmark leaderboard without a reusable method.
- Paper title collides with another work; verify arXiv ID before download.
- No clear relation to physical control or manipulation.

