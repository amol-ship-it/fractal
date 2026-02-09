# Getting Started

A hands-on walkthrough for first-time users. You'll set up the project, understand the core learning system, train a MicroRTS agent, and then train a Zork I agent — all in about 30 minutes.

---

## 1. Setup

### Clone the Repo

```bash
git clone git@github.com:amol-ship-it/fractal.git
cd fractal
```

### Install Python Dependencies

Requires **Python 3.10+** and **pip**.

```bash
pip install -r requirements.txt
```

The key dependency is **numpy**. Everything else (ray, redis, fastapi) is for optional distributed and API features — the game training works without them.

### Verify the Install

```bash
python3 -c "from core import Pattern, PatternEngine, DualMemory, FeedbackLoop; print('Core OK')"
python3 -c "from game import GameEngine, RushAI; print('Game engine OK')"
```

Both should print `OK`. If you see import errors, make sure you're running from the repo root directory.

### Zork-Specific Setup (Do This Now If You Plan to Try Zork)

The Zork AI needs two things that aren't in pip:

**1. dfrotz** — a Z-machine interpreter that runs Zork:

```bash
# macOS
brew install frotz

# Ubuntu / Debian
sudo apt-get install frotz

# Verify it's installed
which dfrotz
```

**2. zork1.z5** — the Zork I game data file. Place it in the repo root:

```bash
ls zork1.z5   # Should show the file
```

If you don't have `zork1.z5`, you can still follow the MicroRTS sections below. The Zork sections require it.

---

## 2. Understanding the Core

Before training any game, it helps to understand what the AI is actually doing. There are no neural networks here. The system learns through four mechanisms working together.

### The Four Pillars

| Pillar | What It Does | Where |
|--------|-------------|-------|
| **Feedback** | When the agent wins or scores, it strengthens the strategies that were used. When it loses, it weakens them. | `core/feedback.py` |
| **Approximability** | Similar game states get mapped to the same "situation key" so the agent can generalize — it doesn't need to see every possible state. | `core/pattern.py` |
| **Composability** | Simple patterns get combined into higher-level concepts. A "control center + develop pieces" pattern might become a "solid opening" concept. | `core/engine.py` |
| **Exploration** | The agent randomly tries new strategies sometimes, to avoid getting stuck on a mediocre approach. This rate decays over training. | `core/feedback.py` |

### How a Decision Happens

Every turn, in every game domain, the same loop runs:

```
1. Read the game state (board position, room description, unit counts)
2. Quantize it into a SITUATION KEY (e.g. "loc_house_s0_i1_d0_e")
3. Look up which STRATEGY has the highest confidence for that situation
4. Execute the strategy to produce a concrete game action
5. After the game ends, update which situation-strategy pairs worked
```

The agent learns a table of **situation → strategy → confidence** mappings. Good results increase confidence; bad results decrease it.

### Try It Yourself

Run the standalone demo to see the four pillars in action:

```bash
python3 system.py
```

This processes sample signals, discovers patterns, applies feedback, and shows how the system learns — no game required.

You can also use the core directly in Python:

```python
from core import Pattern, PatternEngine, DualMemory

# Create the learning system
memory = DualMemory(max_patterns=5000, max_state=500)
engine = PatternEngine(memory)

# Feed it some data and watch it discover patterns
result = engine.process([0.1, 0.3, 0.5, 0.7, 0.9], domain="demo")
print(f"Patterns discovered: {len(result.discovered_patterns)}")
print(f"Predictions made: {len(result.predictions)}")
```

---

## 3. MicroRTS — Your First Game AI

MicroRTS is a real-time strategy game built into the repo. Two players control units on a grid, harvest resources, build structures, and fight. No external dependencies needed — it's pure Python + NumPy.

### See the Game in Action

Watch two built-in bots play against each other:

```bash
python3 -m rts_ai.play --mode demo
```

This runs Rush AI vs Defensive AI and prints an ASCII visualization of the game. You'll see workers harvesting, bases producing units, and armies clashing.

### Train Your First Agent

Now train a pattern-based agent to play against the Rush bot:

```bash
python3 -m rts_ai.train_pattern --episodes 100 --opponent rush --save-path checkpoints_my_first_rts
```

You'll see output like:

```
Starting pattern-based training: 100 episodes
  Opponent: rush
  Map: 8x8
  Initial exploration rate: 30.0%

Episode 10/100 | Win Rate: 40.0% | Patterns: 12 | Bindings: 48 | Explore: 29.1%
Episode 20/100 | Win Rate: 55.0% | Patterns: 22 | Bindings: 63 | Explore: 28.3%
...
```

**What's happening:**
- Each episode is a full game against the Rush bot
- The agent starts with 30% exploration (tries random strategies often)
- As it learns which strategies work in which situations, win rate goes up
- Exploration rate slowly decays — the agent exploits what it's learned
- Patterns and bindings grow as it encounters new situations

### Train Longer with Parallel Processing

For better results, train more episodes using all your CPU cores:

```bash
python3 -m rts_ai.train_pattern --episodes 2000 --opponent rush --parallel --save-path checkpoints_rts_parallel
```

This runs multiple games simultaneously. You should see ~61% win rate after a few thousand episodes.

### Train Against Different Opponents

The agent learns different strategies depending on who it's playing:

```bash
# Economy bot (slow, builds lots of workers)
python3 -m rts_ai.train_pattern --episodes 500 --opponent economy

# Defensive bot (builds barracks and military units)
python3 -m rts_ai.train_pattern --episodes 500 --opponent defensive

# Random bot (easiest — good for initial testing)
python3 -m rts_ai.train_pattern --episodes 500 --opponent random
```

### Resume Training

Training saves checkpoints automatically. Resume from where you left off:

```bash
python3 -m rts_ai.train_pattern --resume checkpoints_my_first_rts --episodes 500
```

### Evaluate the Trained Agent

See how your agent does against all four bots:

```bash
python3 -m rts_ai.play --mode evaluate --checkpoint checkpoints_my_first_rts
```

### Watch the Game in Your Browser

For a visual web-based game viewer:

```bash
# Watch Rush vs Economy in the browser
python3 visualize.py rush economy

# Watch your trained agent play
python3 visualize.py pattern defensive --checkpoint checkpoints_my_first_rts
```

This opens a browser window at `http://localhost:8765` showing the game in real-time.

### Visualize What the Agent Learned

Open the training dashboard to explore the agent's learned strategies:

```bash
python3 -m chess_ai.visualize_chess --checkpoint checkpoints_my_first_rts
```

(Yes, the command says `chess_ai` — the visualizer is domain-agnostic and auto-detects the domain from the checkpoint data.)

This opens a browser dashboard at `http://localhost:8877` with four tabs:
- **Training Timeline** — Win rate, exploration decay, and pattern discovery over time
- **Strategy Map** — Heatmap of which strategy the agent prefers in each situation
- **Strategy Profiles** — The agent's "personality" and strongest strategies
- **Pattern Explorer** — The atomic and composite patterns the agent discovered

### Full CLI Reference (RTS)

```
python3 -m rts_ai.train_pattern [OPTIONS]

  --episodes INT       Number of games to play (default: 500)
  --opponent STR       rush | economy | defensive | random (default: rush)
  --map-size INT       Grid size: 8 or 16 (default: 8)
  --max-ticks INT      Max game length in ticks (default: 2000)
  --log-interval INT   Print stats every N episodes (default: 10)
  --save-path STR      Where to save checkpoints (default: checkpoints_pattern)
  --resume STR         Resume from a checkpoint directory
  --parallel, -p       Use all CPU cores for training
  --workers, -w INT    Number of parallel workers (default: all cores)
  --batch-size INT     Games running at once (default: workers * 2)
  --warmup INT         Sequential episodes before going parallel
```

---

## 4. Zork I — Teaching AI to Play a Text Adventure

This is where things get interesting. Zork I is a classic text adventure from 1980 — the agent reads room descriptions, types commands, manages inventory, solves puzzles, and navigates an underground world. There's no grid, no pixels — just text.

**Prerequisites:** Make sure you completed the [Zork-specific setup](#zork-specific-setup-do-this-now-if-you-plan-to-try-zork) in Section 1.

### How Zork Training Works

The agent communicates with Zork through `dfrotz`, a Z-machine interpreter:

```
Agent reads:  "West of House. You are standing in an open field..."
Agent types:  "north"
Zork replies: "North of House. You are facing the north side..."
Agent types:  "east"
Zork replies: "Behind House. You are behind the white house..."
```

The policy maps each game state (location, inventory, score, visible items) to one of 8 strategies:

| Strategy | What It Does |
|----------|-------------|
| EXPLORE_NEW | Move to rooms the agent hasn't visited yet |
| EXPLORE_KNOWN | Revisit rooms where untried actions remain |
| COLLECT_ITEMS | Pick up visible items |
| USE_ITEM | Try inventory items on objects (puzzle solving) |
| DEPOSIT_TROPHY | Carry treasures back to the trophy case |
| FIGHT | Attack enemies with weapons |
| MANAGE_LIGHT | Turn the lamp on/off in dark areas |
| INTERACT | Examine, open, and read objects |

### Train the Zork Agent

```bash
python3 -m zork_ai.train_zork --game-file zork1.z5 --episodes 300 --save-path checkpoints_my_zork
```

You'll see:

```
Starting Zork pattern-based training: 300 episodes
  Game file: zork1.z5
  Max moves per episode: 400
  Initial exploration rate: 40.0%

Episode 10/300 | Avg Score: 21.4 | Best: 40 | Rooms: 9.1 | Patterns: 13 | Explore: 39.6%
Episode 20/300 | Avg Score: 16.7 | Best: 40 | Rooms: 9.1 | Patterns: 23 | Explore: 39.2%
...
```

**What the numbers mean:**
- **Avg Score**: Average points scored across recent episodes (out of 350 max)
- **Best**: Highest score achieved so far
- **Rooms**: Average number of rooms visited per episode
- **Patterns**: Learned patterns (grows as the agent sees new situations)
- **Explore**: How often the agent tries random strategies (decays over time)

### What the Agent Learns to Do

The agent faces a key challenge: Zork starts you outside a white house, and you **must** find the back window, open it, and climb in to access most of the game. Without guidance, a random agent wanders the forest and never scores.

The system solves this with **early-game navigation** — for the first few moves it follows a deterministic path:

```
West of House → open mailbox → north → North of House → east →
Behind House → open window → enter → Kitchen (score: 10!)
```

Once inside the house, **chokepoint unlocks** fire automatically:
- Kitchen: `take all` (grab sack and bottle), then `west` to Living Room
- Living Room: `move rug`, `open trap door` (access to the underground)
- Attic: `take all` (knife and rope)

After these guided actions, the learned strategies take over.

### Watch the Agent Play

See the agent's decision-making in real-time with color-coded output:

```bash
python3 -m zork_ai.train_zork --game-file zork1.z5 --play --resume checkpoints_my_zork
```

You'll see each move annotated with the strategy name, score changes highlighted, and a summary at the end.

### Train with More Episodes for Better Results

300 episodes takes about 20 seconds. For better performance, train longer:

```bash
python3 -m zork_ai.train_zork --game-file zork1.z5 --episodes 2000 --save-path checkpoints_zork_long
```

Or use parallel training:

```bash
python3 -m zork_ai.train_zork --game-file zork1.z5 --episodes 2000 --parallel --workers 4 --save-path checkpoints_zork_parallel
```

### Visualize Training Progress

```bash
python3 -m chess_ai.visualize_chess --checkpoint checkpoints_my_zork
```

Opens a browser dashboard showing score trends, strategy usage, and learned patterns.

### What Scores to Expect

| Episodes | Avg Score | Scoring Rate | Best Score |
|----------|-----------|-------------|------------|
| 100 | ~8 | ~50% | 25 |
| 300 | ~17 | ~65% | 40 |
| 2000 | ~17 | ~71% | 40 |

"Scoring rate" is the percentage of episodes where the agent scores at least 1 point (meaning it successfully entered the house). The maximum possible score in Zork I is 350.

### Verbose Mode — See What the Agent is Thinking

To see the actual Zork game text during training (for the first 3 episodes):

```bash
python3 -m zork_ai.train_zork --game-file zork1.z5 --episodes 10 --verbose --save-path /tmp/zork_verbose
```

### Full CLI Reference (Zork)

```
python3 -m zork_ai.train_zork [OPTIONS]

Required:
  --game-file PATH     Path to zork1.z5

Options:
  --frotz-path STR     Path to dfrotz binary (default: dfrotz)
  --episodes INT       Training episodes (default: 500)
  --max-moves INT      Max commands per episode (default: 400)
  --log-interval INT   Print stats every N episodes (default: 10)
  --save-path STR      Checkpoint directory (default: checkpoints_zork)
  --resume STR         Resume from a checkpoint directory
  --verbose, -v        Print game text for first 3 episodes
  --play               Watch the AI play one game (no training)
  --parallel, -p       Use all CPU cores for training
  --workers, -w INT    Number of parallel workers (default: all cores)
  --batch-size INT     Games running at once
  --warmup INT         Sequential episodes before going parallel
```

---

## 5. What to Try Next

Now that you've trained both an RTS and a Zork agent, here are some things to explore:

### Chess AI

Train a chess agent (requires `pip install chess` if not already installed):

```bash
pip install chess
python3 -m chess_ai.train_chess --episodes 500 --opponent random
python3 -m chess_ai.train_chess --episodes 500 --opponent minimax1
```

### Compare Strategies Across Domains

Open the visualizer for each domain side by side:

```bash
# Terminal 1
python3 -m chess_ai.visualize_chess --checkpoint checkpoints_my_first_rts --port 8877

# Terminal 2
python3 -m chess_ai.visualize_chess --checkpoint checkpoints_my_zork --port 8878
```

Notice how the same four-pillar framework produces completely different strategies for different games.

### Run the Test Suite

```bash
pytest tests/ -v
```

This runs 1,000+ lines of tests covering the game engine, RTS AI, and chess AI.

### Read the Code

The best places to start reading:

| Want to understand... | Read this file |
|----------------------|----------------|
| How patterns work | `core/pattern.py` (150 lines) |
| How the processing pipeline works | `core/engine.py` |
| How a Zork turn is decided | `zork_ai/zork_policy.py` — `get_command()` method |
| How RTS strategies are selected | `rts_ai/pattern_policy.py` — `get_action()` method |
| How the game engine works | `game/engine.py` — `step()` method |
| How Zork talks to dfrotz | `zork_ai/game_interface.py` — `FrotzInterface` class |

### Checkpoint Files

Every trained agent produces the same four files:

```
checkpoints_*/
  strategy_bindings.json    # Situation → strategy → {wins, losses, confidence}
  patterns.json             # All discovered patterns
  training_stats.json       # Score/win history per episode
  exploration.json          # Current exploration rate
```

These are plain JSON — you can inspect them directly:

```bash
python3 -c "import json; d=json.load(open('checkpoints_my_zork/training_stats.json')); print(f'Episodes: {d[\"episodes_completed\"]}, Best: {d[\"best_score\"]}')"
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'core'"**
Make sure you're running from the repo root directory:
```bash
cd fractal
python3 -m rts_ai.train_pattern --episodes 2
```

**"ModuleNotFoundError: No module named 'numpy'"**
```bash
pip install numpy
```

**"FileNotFoundError: zork1.z5"**
Place the `zork1.z5` file in the repo root. The `--game-file` flag should point to it:
```bash
python3 -m zork_ai.train_zork --game-file ./zork1.z5 --episodes 10
```

**"dfrotz: command not found"**
Install frotz (`brew install frotz` on macOS, `sudo apt install frotz` on Linux). Or specify the path:
```bash
python3 -m zork_ai.train_zork --game-file zork1.z5 --frotz-path /usr/local/bin/dfrotz
```

**"ModuleNotFoundError: No module named 'chess'"**
```bash
pip install chess
```

**Port already in use (visualizer)**
Use a different port:
```bash
python3 -m chess_ai.visualize_chess --checkpoint checkpoints_my_zork --port 9000
```

**Training seems stuck / hanging**
The Zork agent has built-in protections — episodes that hang are killed after a timeout and training continues. If it stops completely, check that `dfrotz` works:
```bash
echo "look" | dfrotz zork1.z5
```
