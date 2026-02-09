# Recursive Learning AI

A pattern-based intelligence system that learns to play games through four fundamental pillars — **Feedback**, **Approximability**, **Composability**, and **Exploration** — without neural networks or gradient descent.

Applied to three game domains: **MicroRTS** (real-time strategy), **Chess** (turn-based strategy), and **Zork I** (text adventure). All three share the same `core/` framework and learn by discovering which situation-strategy pairs produce the best outcomes.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the RTS AI (pattern-based, 500 episodes)
python -m rts_ai.train_pattern --episodes 500 --opponent rush

# 3. Train the Chess AI (500 games vs random opponent)
python -m chess_ai.train_chess --episodes 500 --opponent random

# 4. Train the Zork AI (requires zork1.z5 and dfrotz)
python -m zork_ai.train_zork --game-file zork1.z5 --episodes 500

# 5. Visualize what any agent learned
python -m chess_ai.visualize_chess --checkpoint checkpoints_pattern
```

---

## Table of Contents

- [How It Works](#how-it-works)
- [The Four Pillars](#the-four-pillars)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [MicroRTS AI](#micrortsai)
- [Chess AI](#chess-ai)
- [Zork AI](#zork-ai)
- [Visualizer Dashboard](#visualizer-dashboard)
- [Core Framework](#core-framework)
- [Testing](#testing)
- [Distributed Mode](#distributed-mode)
- [Configuration Reference](#configuration-reference)

---

## How It Works

Every game domain follows the same learning loop:

```
1. OBSERVE the game state
2. QUANTIZE it into a situation key  (Approximability)
3. LOOK UP the best strategy for that situation from learned bindings
4. EXECUTE the strategy to produce a concrete game action
5. RECEIVE feedback from the game outcome  (Feedback)
6. REFINE which situation-strategy pairs work  (Composability + Exploration)
```

The agent learns a mapping from **situations** (discretized game states) to **strategies** (high-level action templates). Each situation-strategy binding tracks wins, losses, and a Bayesian-smoothed confidence score. Over time, the agent converges on effective strategies for each situation.

---

## The Four Pillars

| Pillar | Role | Implementation |
|--------|------|----------------|
| **Feedback Loops** | Learn from outcomes — wins strengthen bindings, losses weaken them | `core/feedback.py` — FeedbackLoop, intrinsic curiosity rewards |
| **Approximability** | Similar states map to the same key, enabling generalization | `core/pattern.py` — Pattern.refine() improves signatures over time |
| **Composability** | Simple patterns combine into higher-level strategic concepts | `core/engine.py` — PatternEngine builds hierarchies (atomic to composite to abstract) |
| **Exploration** | Try novel strategies to discover better approaches | `core/feedback.py` — ExplorationStrategy with epsilon-greedy + decay + minimum floor |

### Dual Memory System

- **Pattern Memory** (Type A): Compressed abstractions — the learned "code" of how to play
- **State Memory** (Type B): Dynamic context — recent situations, linked to patterns

---

## Project Structure

```
recursive_learning_ai/
├── core/                        # Shared four-pillar framework
│   ├── pattern.py               #   Pattern types (atomic, composite, abstract)
│   ├── memory.py                #   Dual memory (PatternMemory + StateMemory)
│   ├── engine.py                #   PatternEngine processing pipeline
│   └── feedback.py              #   FeedbackLoop, ExplorationStrategy
│
├── game/                        # MicroRTS game engine
│   ├── units.py                 #   Unit types (Worker, Light, Heavy, Ranged, Base, Barracks)
│   ├── game_map.py              #   Terrain grid (8x8 or 16x16)
│   ├── game_state.py            #   State representation (20 feature planes)
│   ├── actions.py               #   Action space (move, harvest, produce, attack)
│   ├── engine.py                #   GameEngine + VecGameEnv (vectorized)
│   ├── ai_opponents.py          #   Built-in bots (Random, Rush, Economy, Defensive)
│   ├── renderer.py              #   ASCII visualization
│   ├── visualizer_server.py     #   Web-based game viewer
│   └── visualizer.html          #   Browser UI for game playback
│
├── rts_ai/                      # MicroRTS AI agents
│   ├── encoder.py               #   GameStateEncoder (observation + strategic features)
│   ├── pattern_policy.py        #   Pattern-based policy (no neural network)
│   ├── pattern_agent.py         #   Training loop for pattern agent
│   ├── policy.py                #   GridNetPolicy (CNN-based, for PPO)
│   ├── agent.py                 #   PPOAgent (neural network baseline)
│   ├── knowledge_store.py       #   Persistent strategic knowledge
│   ├── transfer.py              #   Cross-domain transfer learning
│   ├── parallel_worker.py       #   Multi-core episode workers
│   ├── train_pattern.py         #   ENTRY: pattern-based training
│   ├── train.py                 #   ENTRY: PPO training
│   └── play.py                  #   ENTRY: play/evaluate/demo
│
├── chess_ai/                    # Chess AI agent
│   ├── board_encoder.py         #   42-feature board encoding
│   ├── chess_policy.py          #   10 strategies + ChessPatternPolicy
│   ├── chess_agent.py           #   Training loop
│   ├── opponents.py             #   Random, Greedy, Minimax (depth 1-3)
│   ├── train_chess.py           #   ENTRY: chess training
│   ├── visualize_chess.py       #   ENTRY: training dashboard launcher
│   ├── visualizer_server.py     #   HTTP server for dashboard
│   └── visualizer.html          #   Browser dashboard (self-contained)
│
├── zork_ai/                     # Zork I AI agent
│   ├── game_interface.py        #   FrotzInterface (dfrotz subprocess control)
│   ├── text_parser.py           #   Parse game responses (exits, items, death)
│   ├── text_encoder.py          #   Text-to-feature encoding
│   ├── zork_policy.py           #   8 strategies + early-game navigation + chokepoint unlocks
│   ├── zork_agent.py            #   Training loop (sequential + parallel)
│   ├── parallel_worker.py       #   Multi-core episode workers
│   ├── train_zork.py            #   ENTRY: zork training + play mode
│   └── visualize_zork.py        #   Training visualizer launcher
│
├── distributed/                 # Distributed cluster system (optional)
│   ├── cluster.py               #   RecursiveLearningCluster API
│   ├── coordinator.py           #   Cluster brain
│   ├── worker.py                #   Processing workers
│   ├── storage.py               #   Redis-backed storage
│   └── config.py                #   ClusterConfig, NodeRole
│
├── tests/                       # Test suites
│   ├── test_game.py             #   MicroRTS engine tests (399 lines)
│   ├── test_rts_ai.py           #   RTS AI tests (292 lines)
│   └── test_chess_ai.py         #   Chess AI tests (356 lines)
│
├── system.py                    # Standalone RecursiveLearningAI class + demo
├── cli.py                       # CLI tool (start, status, benchmark, demo)
├── visualize.py                 # MicroRTS game visualizer launcher
├── requirements.txt             # Python dependencies
├── VISUALIZER.md                # Dashboard documentation
└── DEPLOYMENT.md                # Distributed deployment guide
```

---

## Installation

### Requirements

- **Python 3.10+**
- **numpy** (core dependency)
- **chess** (for Chess AI only — `pip install chess`)
- **dfrotz** (for Zork AI only — Z-machine interpreter)
- **zork1.z5** (for Zork AI only — Zork I game data file)

<<<<<<< nice-elion
### Install

```bash
git clone git@github.com:amol-ship-it/fractal.git
cd fractal
=======
# Install dependencies
>>>>>>> main
pip install -r requirements.txt
```

### Verify

```bash
python3 -c "from core import Pattern, PatternEngine, DualMemory, FeedbackLoop; print('Core OK')"
python3 -c "from game import GameEngine, RushAI; print('Game OK')"
python3 -c "from rts_ai import PatternAgent; print('RTS AI OK')"
python3 -c "from chess_ai import ChessPatternAgent; print('Chess AI OK')"
python3 -c "from zork_ai import ZorkPatternAgent; print('Zork AI OK')"
```

### Zork-Specific Setup

The Zork AI requires `dfrotz` (the Frotz Z-machine interpreter) and `zork1.z5`:

```bash
# macOS
brew install frotz

# Ubuntu/Debian
sudo apt-get install frotz

# Verify dfrotz is available
dfrotz --version
```

Place `zork1.z5` in the project root directory.

---

## MicroRTS AI

A real-time strategy game where two players control units on a grid map, harvesting resources, building structures, and fighting.

### Game Features

- **6 unit types**: Worker, Light, Heavy, Ranged, Base, Barracks
- **6 action types**: Move, Harvest, Return, Produce, Attack, Idle
- **Map sizes**: 8x8 (default) or 16x16
- **4 built-in opponents**: Random, Rush, Economy, Defensive
- **20 feature planes** per observation (unit types, resources, terrain)

### Training (Pattern-Based)

```bash
# Basic training (500 episodes vs Rush AI)
python -m rts_ai.train_pattern --episodes 500 --opponent rush

# Parallel training (uses all CPU cores)
python -m rts_ai.train_pattern --episodes 5000 --parallel --workers 4

# Resume from checkpoint
python -m rts_ai.train_pattern --resume checkpoints_pattern --episodes 500

# Train against different opponents
python -m rts_ai.train_pattern --episodes 1000 --opponent economy
python -m rts_ai.train_pattern --episodes 1000 --opponent defensive
```

**Full CLI options:**
```
python -m rts_ai.train_pattern [OPTIONS]

  --episodes INT       Training episodes (default: 500)
  --opponent STR       rush | economy | defensive | random (default: rush)
  --map-size INT       8 or 16 (default: 8)
  --max-ticks INT      Game length limit (default: 2000)
  --log-interval INT   Print stats every N episodes (default: 10)
  --save-path STR      Checkpoint directory (default: checkpoints_pattern)
  --resume STR         Resume from checkpoint directory
  --parallel, -p       Enable multi-core training
  --workers, -w INT    Worker processes (default: cpu_count)
  --batch-size INT     Concurrent episodes per batch (default: workers * 2)
  --warmup INT         Sequential episodes before going parallel
```

### Training (PPO Neural Network)

A CNN-based PPO baseline for comparison:

```bash
python -m rts_ai.train --timesteps 100000 --opponent rush
python -m rts_ai.train --num-envs 8 --n-steps 128 --opponent curriculum
```

**Full CLI options:**
```
python -m rts_ai.train [OPTIONS]

  --timesteps INT      Total training steps (default: 100000)
  --num-envs INT       Parallel environments (default: 4)
  --map-size INT       8 or 16 (default: 8)
  --max-ticks INT      Game length (default: 2000)
  --opponent STR       random | rush | economy | defensive | curriculum (default: rush)
  --n-steps INT        Rollout length (default: 128)
  --log-interval INT   Print every N updates (default: 10)
  --save-path STR      Checkpoint directory (default: checkpoints)
  --resume STR         Resume from checkpoint
```

### Play / Evaluate / Demo

```bash
# Watch a demo game
python -m rts_ai.play --mode demo

# Evaluate trained agent against all bots
python -m rts_ai.play --mode evaluate --checkpoint checkpoints_pattern

# Watch trained agent play
python -m rts_ai.play --mode watch --checkpoint checkpoints_pattern

# Cross-domain transfer demo
python -m rts_ai.play --mode transfer-demo
```

### Visualize Games in Browser

```bash
# Watch Rush vs Economy
python visualize.py rush economy

# Watch trained pattern agent vs Defensive AI
python visualize.py pattern defensive --checkpoint checkpoints_pattern

# Larger map
python visualize.py rush economy --size 16 --ticks 2000

# Custom port
python visualize.py rush economy --port 9000
```

**Full CLI options:**
```
python visualize.py [P0] [P1] [OPTIONS]

  P0, P1               AI type: rush | economy | defensive | random | ppo | pattern
  --size INT            Map size: 8 | 12 | 16 (default: 8)
  --ticks INT           Max game ticks (default: 1000)
  --port INT            HTTP server port (default: 8765)
  --checkpoint PATH     Trained agent checkpoint directory
  --no-browser          Don't auto-open browser
```

### Performance

The pattern-based agent achieves ~61% win rate against Rush AI with parallel training.

---

## Chess AI

A pattern-matching chess agent that learns 10 high-level strategies and maps board positions to the best strategy.

### Strategies

| # | Strategy | Description |
|---|----------|-------------|
| 0 | DEVELOP | Piece development in the opening |
| 1 | CONTROL_CENTER | Central square dominance |
| 2 | ATTACK_KING | King-side offensive |
| 3 | TRADE_PIECES | Simplify when ahead in material |
| 4 | PUSH_PAWNS | Pawn promotion race |
| 5 | DEFEND | Defensive consolidation |
| 6 | CASTLE | King safety via castling |
| 7 | ENDGAME_PUSH | Endgame breakthrough |
| 8 | QUIET_MOVE | Positional maneuvering |
| 9 | PROPHYLAXIS | Preventive moves |

### Board Encoding

Each position is encoded as a 42-element feature vector capturing:
- Piece counts (6 per side)
- Piece position concentration
- King safety metrics (zone attacks, pawn shields)
- Pawn structure (passed, isolated, doubled, hanging pawns)
- Material balance

### Training

```bash
# Train vs random opponent (easiest)
python -m chess_ai.train_chess --episodes 500 --opponent random

# Train vs greedy opponent (captures everything)
python -m chess_ai.train_chess --episodes 1000 --opponent greedy

# Train vs minimax (depth 1, 2, or 3)
python -m chess_ai.train_chess --episodes 1000 --opponent minimax1
python -m chess_ai.train_chess --episodes 1000 --opponent minimax2

# Play as black
python -m chess_ai.train_chess --episodes 500 --opponent random --color black

# Resume training
python -m chess_ai.train_chess --resume checkpoints_chess --episodes 500
```

**Full CLI options:**
```
python -m chess_ai.train_chess [OPTIONS]

  --episodes INT       Training games (default: 500)
  --opponent STR       random | greedy | minimax1 | minimax2 | minimax3 (default: random)
  --color STR          white | black (default: white)
  --max-moves INT      Half-moves per game (default: 200)
  --log-interval INT   Print every N episodes (default: 10)
  --save-path STR      Checkpoint directory (default: checkpoints_chess)
  --resume STR         Resume from checkpoint
```

### Visualize Training

```bash
# Open dashboard for chess training
python -m chess_ai.visualize_chess

# Open dashboard for any checkpoint
python -m chess_ai.visualize_chess --checkpoint checkpoints_chess --port 9000
```

---

## Zork AI

A text adventure agent that learns to play Zork I by interacting with a Z-machine interpreter (dfrotz). The agent reads room descriptions, manages inventory, solves puzzles, and navigates a complex underground world.

### Strategies

| # | Strategy | Description |
|---|----------|-------------|
| 0 | EXPLORE_NEW | Move to unvisited rooms via untried exits |
| 1 | EXPLORE_KNOWN | Revisit rooms with untried actions |
| 2 | COLLECT_ITEMS | Pick up visible items |
| 3 | USE_ITEM | Try items on objects (puzzle solving) |
| 4 | DEPOSIT_TROPHY | Bring treasures to the trophy case |
| 5 | FIGHT | Attack enemies with weapons |
| 6 | MANAGE_LIGHT | Handle the lamp in dark areas |
| 7 | INTERACT | Examine, open, read objects |

### Key Features

- **Early-game navigation**: Deterministic path from West of House through the house circuit (north to North of House, east to Behind House, open window, enter) to reliably reach the Kitchen
- **Chokepoint unlocks**: Automatic one-time actions at key gates (open window, move rug, open trap door, take all in Kitchen/Attic) that fire from ALL strategies
- **Context-aware strategy filtering**: Only strategies that can do something useful in the current state are considered (e.g., FIGHT is only viable when enemies are present)
- **3-rule credit assignment**: (1) Direct score credit for strategies causing score increases, (2) Episode-level credit proportional to final score, (3) Exploration proxy using rooms visited for zero-score episodes
- **Non-blocking I/O**: Uses `select.select()` to prevent training hangs when dfrotz stalls
- **Per-episode crash protection**: Failed episodes are logged and skipped without aborting training

### Training

```bash
# Basic training (500 episodes)
python -m zork_ai.train_zork --game-file zork1.z5 --episodes 500

# Parallel training
python -m zork_ai.train_zork --game-file zork1.z5 --episodes 5000 --parallel --workers 4

# Resume training from checkpoint
python -m zork_ai.train_zork --game-file zork1.z5 --resume checkpoints_zork --episodes 1000

# Verbose mode (shows game output for first 3 episodes)
python -m zork_ai.train_zork --game-file zork1.z5 --episodes 100 --verbose
```

**Full CLI options:**
```
python -m zork_ai.train_zork [OPTIONS]

Required:
  --game-file PATH     Path to zork1.z5

Options:
  --frotz-path STR     Path to dfrotz binary (default: dfrotz)
  --episodes INT       Training episodes (default: 500)
  --max-moves INT      Moves per episode (default: 400)
  --log-interval INT   Print every N episodes (default: 10)
  --save-path STR      Checkpoint directory (default: checkpoints_zork)
  --resume STR         Resume from checkpoint
  --verbose, -v        Print game text for first 3 episodes
  --play               Watch one episode (visual mode, no training)
  --parallel, -p       Enable multi-core training
  --workers, -w INT    Worker processes (default: cpu_count)
  --batch-size INT     Concurrent episodes per batch
  --warmup INT         Sequential warmup before parallel
```

### Watch the Agent Play

```bash
# Visual playthrough with color-coded strategy decisions
python -m zork_ai.train_zork --game-file zork1.z5 --play --resume checkpoints_zork_v4
```

This shows real-time gameplay with:
- Room descriptions and location changes
- Strategy chosen for each move (color-coded)
- Score change highlights
- Episode summary with strategy usage breakdown

### Performance

| Metric | Before Navigation Fix | After |
|--------|----------------------|-------|
| Scoring rate | 30% | **71%** |
| Average score | 4.6 / 350 | **16.6 / 350** |
| Best score | 40 / 350 | 40 / 350 |
| Most common outcome | 0 (70%) | **25 (48%)** |

The early-game navigation ensures the agent reliably enters the house, while chokepoint unlocks open up the underground.

---

## Visualizer Dashboard

An interactive browser dashboard for exploring what any trained agent has learned. Works with all three game domains — auto-detects the domain from checkpoint data.

### Launch

```bash
# Chess training results
python -m chess_ai.visualize_chess --checkpoint checkpoints_chess

# RTS training results
python -m chess_ai.visualize_chess --checkpoint checkpoints_pattern

# Zork training results
python -m chess_ai.visualize_chess --checkpoint checkpoints_zork_v4

# Custom port, no auto-open
python -m chess_ai.visualize_chess --checkpoint checkpoints_zork_v4 --port 9000 --no-browser
```

### Dashboard Tabs

1. **Training Timeline** — Win/loss rate, episode duration, exploration decay, pattern discovery curves
2. **Strategy Map** — Heatmap of situation-to-strategy confidence with domain-specific filters
3. **Strategy Profiles** — AI personality analysis, strategy usage radar, per-strategy breakdowns
4. **Pattern Explorer** — Pattern type distribution, confidence histogram, top patterns by activation

### Checkpoint Format

All domains produce the same checkpoint structure:

```
checkpoints_*/
  strategy_bindings.json    # Situation key -> strategy -> {wins, losses, confidence}
  patterns.json             # Discovered patterns (atomic + composite)
  training_stats.json       # Per-episode score/win history
  exploration.json          # Current exploration rate
```

See [VISUALIZER.md](VISUALIZER.md) for full dashboard documentation.

---

## Core Framework

The `core/` module provides the domain-agnostic four-pillar system that all game agents share.

### Pattern (`core/pattern.py`)

The atomic unit of knowledge. Three types form a hierarchy:

- **Atomic** — Base-level features extracted from raw input
- **Composite** — Combinations of atomic patterns (higher-level concepts)
- **Abstract** — Cross-domain generalizations

```python
from core import Pattern

# Create an atomic pattern from features
p = Pattern.create_atomic([0.1, 0.3, 0.5], domain="chess_situation")

# Combine patterns
composite = Pattern.create_composite([p1, p2, p3], domain="strategy")

# Refine with feedback
p.refine(feedback=0.8)  # Positive outcome
p.refine(feedback=0.2)  # Negative outcome

# Compare patterns
similarity = p1.similarity(p2)  # Cosine similarity [0, 1]
```

### PatternEngine (`core/engine.py`)

Bottom-up processing pipeline:

```
RAW INPUT → EDGE DETECTION → CLUSTERING → PATTERN DISCOVERY → COMPOSITION → PREDICTION
            (subtraction)    (division)   (similarity match)   (hierarchy)   (forecast)
```

```python
from core import PatternEngine, DualMemory

memory = DualMemory(max_patterns=5000, max_state=500)
engine = PatternEngine(memory)

# Process features and discover patterns
result = engine.process([0.1, 0.3, 0.5, 0.7], domain="game_state")
print(f"Discovered: {result.discovered_patterns}")
print(f"Predictions: {result.predictions}")

# Find similar patterns
matches = engine.query([0.2, 0.4, 0.6], domain="game_state")
```

### FeedbackLoop (`core/feedback.py`)

Feedback propagation and exploration control:

```python
from core import FeedbackLoop, FeedbackSignal, FeedbackType, ExplorationStrategy

loop = FeedbackLoop(learning_rate=0.1, discount_factor=0.99)
explorer = ExplorationStrategy(exploration_rate=0.4)

# Should the agent explore or exploit?
if explorer.should_explore():
    action = random_action()
else:
    action = best_known_action()

# Apply feedback after outcome
signal = FeedbackSignal(
    signal_type=FeedbackType.EXTRINSIC,
    value=0.8,  # [-1, 1] scale
    target_pattern_ids=["pattern_abc", "pattern_def"],
    context={"score": 25}
)
loop.apply_feedback(signal, memory.patterns.patterns)

# Decay exploration over time
explorer.decay_exploration(0.999)
```

### Standalone System (`system.py`)

Use the core framework directly without any game domain:

```python
from system import RecursiveLearningAI

ai = RecursiveLearningAI({'learning_rate': 0.1, 'exploration_rate': 0.2})

# Process input data
result = ai.perceive([0.1, 0.3, 0.5, 0.7], domain="sensor")

# Provide feedback
ai.receive_feedback(result['episode_id'], feedback_value=0.8)

# Query for similar patterns
matches = ai.query([0.2, 0.4, 0.6])

# System stats
print(ai.introspect())
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run by domain
pytest tests/test_game.py -v        # MicroRTS engine (units, map, actions, AI)
pytest tests/test_rts_ai.py -v      # RTS AI (encoder, policy, agent, knowledge, transfer)
pytest tests/test_chess_ai.py -v    # Chess AI (encoder, strategy, policy, opponents, agent)

# Run a specific test class
pytest tests/test_rts_ai.py::TestGameStateEncoder -v

# With coverage
pytest tests/ -v --cov=core --cov=game --cov=rts_ai --cov=chess_ai
```

### Test Coverage

| File | Tests |
|------|-------|
| `test_game.py` | Units, map, state encoding, action masking, engine mechanics, 4 AI opponents, vectorized env |
| `test_rts_ai.py` | State encoder, CNN policy, PPO agent, rollout buffer, knowledge store, transfer bridge |
| `test_chess_ai.py` | Board encoder, strategy bindings, pattern policy, move generation, 3 opponents, training loop |

---

## Distributed Mode

For large-scale training or pattern processing across multiple machines.

### Single Machine

```bash
python cli.py start --workers 8
python cli.py demo
python cli.py benchmark --signals 1000 --workers 4
```

### Multi-Machine Cluster

```bash
# Machine 1: Coordinator + Redis
redis-server
python cli.py start --role coordinator --redis localhost

# Machine 2+: Workers
python cli.py start --role worker --coordinator 192.168.1.100 --redis 192.168.1.100
```

### Docker / Kubernetes

See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker Compose, Kubernetes, and production deployment instructions.

### CLI Commands

```
python cli.py [COMMAND] [OPTIONS]

Commands:
  start       Start cluster node
  status      Show cluster status
  process     Process data through the pipeline
  query       Search for matching patterns
  scale       Scale worker count
  benchmark   Run performance benchmark
  demo        Run interactive demo

Global Options:
  --verbose, -v    Verbose output
```

---

## Configuration Reference

### Core Engine Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `edge_threshold` | 0.1 | Minimum gradient for edge detection |
| `cluster_threshold` | 0.7 | Similarity threshold for clustering |
| `learning_rate` | 0.1 | Pattern update rate |
| `exploration_rate` | 0.2 - 0.4 | Initial exploration probability (domain-dependent) |
| `exploration_decay` | 0.998 - 0.999 | Per-episode exploration decay rate |
| `min_exploration` | 0.15 | Exploration floor (Zork) |

### Strategy Binding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pseudo_count` | 2 | Bayesian smoothing for confidence calculation |
| Confidence formula | `(wins + 1) / (wins + losses + 2)` | Bayesian-smoothed win rate |

### Environment Variables (Distributed)

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_WORKERS` | 4 | Workers per node |
| `REDIS_HOST` | localhost | Redis server address |
| `MIN_WORKERS` | 2 | Minimum workers (auto-scale) |
| `MAX_WORKERS` | 50 | Maximum workers (auto-scale) |

---

## Research Foundation

The four-pillar architecture draws from:

- **General Learning Naturalism** — Intelligence as pattern discovery from experience
- **Perceptual Magnet Effect** — Learned patterns warp future perception
- **Hierarchical Aggregation** — Building abstractions from raw input (atoms to composites)
- **Swarm Intelligence** — Independent agents self-organizing through shared feedback

---

## License

MIT License — Free for research and commercial use.
