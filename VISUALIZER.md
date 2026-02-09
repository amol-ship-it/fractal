# Pattern Visualizer

Interactive dashboard for exploring what the Recursive Learning AI has learned during training. Works with any game domain — Chess, MicroRTS, or future games. The visualizer auto-detects the domain from checkpoint data and adapts its interface accordingly.

---

## Quick Start

```bash
# View chess training results (default)
python -m chess_ai.visualize_chess

# View RTS training results
python -m chess_ai.visualize_chess --checkpoint checkpoints_pattern

# Custom port
python -m chess_ai.visualize_chess --port 9000

# Server only (no auto-open browser)
python -m chess_ai.visualize_chess --no-browser
```

The visualizer opens automatically at `http://localhost:8877`.

---

## Requirements

- Python 3.10+
- A checkpoint directory containing training data (produced by `train_chess.py` or the RTS training loop)
- A modern web browser

No additional Python packages are needed beyond the standard library.

---

## Checkpoint Format

The visualizer reads four JSON files from the checkpoint directory. These files are produced automatically during training and share an identical structure across all game domains.

```
checkpoints_chess/          # or checkpoints_pattern/ for RTS
  strategy_bindings.json    # Situation -> strategy win/loss/confidence data
  patterns.json             # Discovered patterns (atomic + composite)
  training_stats.json       # Episode-by-episode training history
  exploration.json          # Current exploration rate
```

| File | Contents |
|------|----------|
| `strategy_bindings.json` | Maps each situation key to a set of strategies with win count, loss count, times used, and confidence |
| `patterns.json` | Every pattern the AI discovered — type (atomic/composite), confidence, activation count, and domain tags |
| `training_stats.json` | Per-episode records: win/loss/draw outcomes, game durations, pattern counts over time |
| `exploration.json` | The AI's current exploration rate (how often it tries random strategies vs. exploiting known ones) |

---

## Domain Detection

The visualizer identifies the game domain by inspecting the format of situation keys in `strategy_bindings.json`:

| Domain | Key Pattern | Example |
|--------|-------------|---------|
| Chess | `mat{material}_{phase}_{safety}_{dev}_{center}` | `mat=_opn_saf_hi_ceq` |
| MicroRTS | `w{workers}_b{barracks}_c{combat}_r{resources}_{phase}` | `w2_b1_c0_r0_m` |
| Generic | Anything else | Displayed raw with underscore-split components |

Once the domain is detected, the dashboard adapts:
- Header badge shows the domain name
- Situation keys are decoded into human-readable descriptions
- Strategy names, colors, and descriptions match the domain
- Heatmap filters change to domain-relevant categories
- Training stats handle domain-specific fields (e.g., draw rate for chess, binary win/loss for RTS)

---

## Dashboard Tabs

### Tab 1: Training Timeline

Four time-series charts showing the AI's learning progression across all training episodes.

**Win / Loss Rate (Rolling 20-episode)**
Smoothed win and loss rates over time. For chess, a draw rate line is also shown. Reveals whether the AI improved over the course of training.

**Episode Duration**
Scatter plot of how long each game lasted (moves for chess, ticks for RTS). Patterns in duration often indicate the AI learning to win faster or avoid prolonged losses.

**Exploration Rate Decay**
Shows how the exploration rate decreased over training. High early exploration means the AI tried many strategies; low late exploration means it settled on what works.

**Pattern Discovery (Cumulative)**
Cumulative count of unique patterns discovered. A steep curve means rapid learning; a plateau means the AI has mapped most of the pattern space.

### Tab 2: Strategy Map

An interactive heatmap showing every situation the AI encountered and the confidence it has in each strategy for that situation.

**Reading the heatmap:**
- Each row is a situation (decoded into human-readable form)
- Each column is a strategy
- Cell color indicates confidence: red (low) to yellow (medium) to green (high)
- Cell text shows the win-rate percentage
- Empty cells mean that strategy was never tried in that situation

**Filters:**
- Chess: filter by game phase (Opening / Middlegame / Endgame) and material balance (Losing badly / Losing / Equal / Winning / Winning big)
- RTS: filter by game phase (Early / Mid / Late), worker count, and combat intensity
- Sort by total usage, best confidence, or alphabetically

**Clicking a cell** opens a detail panel showing exact wins, losses, times used, and confidence for that strategy-situation pair.

### Tab 3: Strategy Profiles

A personality profile of the trained AI, generated from its strategy usage data.

**AI Personality**
- Playing style summary identifying the AI's most-used strategies
- Key insights listing situations where the AI learned a strategy works 100% of the time (or very close to it)
- Endgame/late-game focus showing which strategies the AI prefers in the final phase

**Strategy Usage Radar**
Radar chart showing relative usage of each strategy — reveals whether the AI is balanced or specialized.

**Strategy Comparison**
Horizontal bar chart comparing average confidence across all strategies.

**Individual Strategy Profiles**
Cards for each strategy showing its description, average confidence, total times used, and the situation where it performs best.

### Tab 4: Pattern Explorer

Deep dive into the patterns the AI has discovered through the Composability pillar.

**Pattern Type Breakdown**
Pie chart showing the ratio of atomic (base-level) vs. composite (combined) vs. abstract patterns. Composite patterns are built by combining atomic ones and represent higher-level strategic concepts.

**Confidence Distribution**
Histogram of pattern confidence scores. A right-skewed distribution means most patterns are well-validated; a left-skewed distribution means many patterns are still uncertain.

**Cross-Domain Pattern Distribution**
When patterns span multiple domains (e.g., `chess_situation` and `rts_situation`), a pie chart shows the breakdown. This visualizes how the four pillars of learning enable cross-domain pattern sharing.

**Top Patterns by Activation Count**
Ranked list of the most frequently activated patterns. Each entry shows:
- Pattern ID (hash)
- Type (ATOMIC / COMPOSITE)
- Domain tag
- Activation count bar
- Confidence percentage

**Clicking a pattern** opens a detail panel showing its full metadata: type, confidence, activation count, domain, component count (for composite patterns), and creation context.

---

## Architecture

```
┌──────────────────────┐       ┌──────────────────────┐
│  visualize_chess.py  │──────►│  visualizer_server.py │
│  (launcher)          │       │  (HTTP server)        │
│  --checkpoint PATH   │       │  /api/data → JSON     │
│  --port PORT         │       │  /        → HTML      │
└──────────────────────┘       └───────────┬──────────┘
                                           │
                               ┌───────────▼──────────┐
                               │   visualizer.html     │
                               │   (dashboard)         │
                               │   - Domain detection  │
                               │   - Canvas charts     │
                               │   - Interactive heatmap│
                               │   - Pattern explorer  │
                               └───────────────────────┘
                                           │
                               ┌───────────▼──────────┐
                               │  Checkpoint Directory │
                               │  - strategy_bindings  │
                               │  - patterns           │
                               │  - training_stats     │
                               │  - exploration        │
                               └───────────────────────┘
```

**visualize_chess.py** — Launcher script. Parses CLI arguments, starts the server, and opens the browser.

**visualizer_server.py** — Lightweight HTTP server (standard library only). Serves the HTML dashboard at `/` and checkpoint data as JSON at `/api/data`. Auto-detects the game domain from situation key format and includes it in the API response.

**visualizer.html** — Self-contained single-file dashboard. All CSS, JavaScript, and chart rendering is embedded (no external dependencies). Uses HTML5 Canvas for all visualizations.

---

## API Endpoints

The server exposes a simple REST API (useful for building custom tools or programmatic access):

| Endpoint | Returns |
|----------|---------|
| `GET /` | The HTML dashboard |
| `GET /api/data` | All checkpoint data in a single JSON blob (strategy bindings, patterns, stats, exploration rate, detected domain, strategy names, metadata) |
| `GET /api/bindings` | Raw `strategy_bindings.json` |
| `GET /api/patterns` | Raw `patterns.json` |
| `GET /api/stats` | Raw `training_stats.json` |

Example:
```bash
# Get all data
curl http://localhost:8877/api/data | python -m json.tool

# Get just the patterns
curl http://localhost:8877/api/patterns | python -m json.tool
```

---

## Command Reference

```
python -m chess_ai.visualize_chess [OPTIONS]

Options:
  --checkpoint PATH   Checkpoint directory to visualize (default: checkpoints_chess)
  --port PORT         HTTP server port (default: 8877)
  --no-browser        Don't open the browser automatically
```

**Examples:**

```bash
# Chess AI training results
python -m chess_ai.visualize_chess --checkpoint checkpoints_chess

# RTS AI training results
python -m chess_ai.visualize_chess --checkpoint checkpoints_pattern

# Any future domain's checkpoint
python -m chess_ai.visualize_chess --checkpoint checkpoints_my_game

# Run on a different port (useful if 8877 is in use)
python -m chess_ai.visualize_chess --port 9999
```

---

## Adding a New Domain

The visualizer supports any game that uses the four pillars checkpoint format. To add full domain support (decoded situation keys, named strategies, custom filters):

1. **Domain detection** — Add a regex match for your situation key format in `visualizer_server.py`'s `detect_domain()` function and in `visualizer.html`'s `detectDomain()` function.

2. **Strategy names** — Add your strategy list to `DOMAIN_STRATEGIES` in `visualizer_server.py` and to `DOMAIN_STRATEGIES` in `visualizer.html`.

3. **Situation key decoder** — Add a `decodeMyDomainKey()` function in `visualizer.html` that converts raw keys to human-readable descriptions.

4. **Heatmap filters** — Add domain-specific filter definitions in `buildHeatmapFilters()` and filter logic in `filterRow()`.

Without these additions, any domain will still work — it just displays raw situation keys and generic strategy names ("Strategy 0", "Strategy 1", etc.).

---

## Troubleshooting

**"Error: checkpoint directory not found"**
Make sure the checkpoint directory exists and contains the four JSON files. Run training first if needed:
```bash
# Chess
python -m chess_ai.train_chess

# RTS
python -m rts_ai.train
```

**Dashboard shows empty charts**
The checkpoint files may be empty or malformed. Verify with:
```bash
python -c "import json; print(len(json.load(open('checkpoints_chess/patterns.json'))))"
```

**Port already in use**
Another process is using port 8877. Either stop it or use a different port:
```bash
python -m chess_ai.visualize_chess --port 9000
```

**Browser doesn't open automatically**
Some environments (SSH, containers) can't open a browser. Use `--no-browser` and navigate manually:
```bash
python -m chess_ai.visualize_chess --no-browser
# Then open http://localhost:8877 in your browser
```
