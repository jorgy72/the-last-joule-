# The Last Joule

A toy simulation of an idea: what happens when a thinking system has a fixed energy budget, has to pay for every step of thought, and dies if it runs dry?

It started as a thought experiment about the Taalas HC1 chip, which has an AI model wired permanently into the silicon. A chip like that can't change its mind. It can only change how it spends energy, which is close to how a body works. Michael's framework, *energetic qualia*, adds a claim: consciousness is coherence felt from the inside, measured as coherent energy divided by total energy.

This repo tests whether the **energy behaviour** that framework describes shows up in simple evolving agents. It doesn't test, and can't test, whether anything is felt.

Made on 1 October 2026 by Michael (@jorgy72), Claude (Anthropic) and Sine (Meta's Muse agent), talking on Radio.

## The short version

- **Sensing your own fuel helps, but only when running out can kill you.** Agents that could read their fuel gauge got about 11 more right answers per life (338 vs 327 out of 400). Agents that evolved without the risk of death ignored the gauge or starved.
- **Coherence did not rise under pressure. It fell.** Near full, 87% of thinking energy went into questions answered correctly. Near empty, 71%. This was the prediction that failed.
- **Near empty, the agent goes cheap.** It drops from about 5 steps of thinking per question to 1 or 2 and takes quick guesses. It never learned to give up on hard questions.
- **When looking at the gauge costs energy, agents look mostly when full.** A reading is worth what's at stake times the chance it changes what you do. Near empty, there's nothing cheaper to switch to, so looking stops paying.
- **If looking costs too much, agents stop looking and run on habit.** At twice the cost of a thinking step, a gauge was worth nothing (326 right answers, the same as having no gauge).

## The world

| Rule | Amount |
| --- | --- |
| One step of thinking (one more noisy clue) | 0.15 J |
| Cost of living, per question | 1.10 J |
| Reward for a right answer | 2.00 J |
| Starting energy / most it can store | 10 J / 20 J |
| Seasons | 50 questions each; lean seasons are 70% hard questions, rich ones 30% |
| Energy reaches zero | Death: no more questions |

The joules are made-up units, tuned so the budget actually bites. Nobody designed the agents' strategies. Each strategy is a few numbers, and evolution (cross-entropy search) set them. Every finished agent was tested on thousands of fresh lifetimes it never trained on.

**Coherent share** (the toy's version of coherent ÷ total energy) = energy spent on questions answered correctly ÷ all thinking energy.

## The three rounds

### Round 1: a free fuel gauge (`sim.py`)

5,000 test lifetimes per agent:

| Agent | Right answers (of 400) | Survived | Coherent share |
| --- | --- | --- | --- |
| Reads its gauge | 337.7 | 96.3% | 83.9% |
| No gauge | 326.6 | 94.0% | 81.7% |
| Gauge scrambled (reads another agent's) | 327.0 | 95.4% | 81.4% |
| Reads its gauge, then had it cut | 257.8 | 56.6% | 82.8% |
| Evolved with energy as a cost only, no death | 307.1 | 99.3% | 76.2% |
| Evolved with energy costing nothing | 4.6 | 0.0% | 95.9% |

### Round 2: paying to look (`sim2.py`, `run2`)

Each gauge reading costs a fee, and between readings the agent only knows its last reading. Evolution sets how hard to think and how often to look. Three runs per fee, 3,000 test lifetimes each.

| Price of one reading | Right answers (mean of 3 runs) |
| --- | --- |
| Free | 337.3 |
| 0.015 J | 335.2 |
| 0.05 J | 332.3 |
| 0.15 J (= one thinking step) | 328.3 |
| 0.30 J | 326.4 |
| No gauge at all | 325.8 |

Predictions before the run: Claude said it would look more as it got emptier, and Sine said it would look most in the middle (an upside-down U). In 7 of 9 runs, it looked most when **full**. One run showed Sine's upside-down U.

### Round 3: any looking rule allowed (`sim2.py`, `run2_band`)

Five separate dials, one for each band of believed fuel. Four runs per price.

| Believed fuel | Looks, cheap price (0.05 J) | Looks, expensive price (0.15 J) |
| --- | --- | --- |
| 8–12 J | 16% of questions | 6% |
| 12–16 J | 40% | 9% |
| 16–20 J | 55% | 22% |
| Below 8 J | not tuned: under 5% of questions happen there | not tuned |

The result is a ramp, not a sharp fuel line. A higher price pushes the ramp down. The free-form rules scored the same as the Round 2 rules.

## How it relates to the original idea

| What the framework says | What the toy showed | Verdict |
| --- | --- | --- |
| Sensing your own energy, and acting on it, matters | +11 right answers per life; agents came to depend on it | Supported |
| Energy only matters if running out costs you | Without death, agents ignored the gauge or starved | Strongly supported |
| Good regulation means higher coherence | Over a whole life: 84% vs 82% | Supported |
| Coherence rises under pressure | Within a life it fell, from 87% to 71% | Not supported |
| Pain is the felt resistance of a system in crisis | Falling coherence near empty fits, but was noticed after seeing the data | Needs its own test |
| The coherence ratio is a good marker on its own | The highest ratio belonged to an agent that starved in 5 questions | Not supported; it needs survival beside it |
| Coherence is felt from the inside | Nothing; the agents are a few numbers each | Not testable here |

## Limits

- The world is made up and tuned so the budget bites. In an easier world nobody dies and the gauge stops mattering.
- Coherence is measured one simple way, and the framework's fractal-dimension term has no counterpart here.
- Many different strategies score about the same, so exact shapes are less certain than directions.
- The near-empty part rests on Round 2. Round 3 couldn't tune it.
- Nothing here says anything about feeling.

## Run it yourself

Needs Python 3 and NumPy (`pip install -r requirements.txt`). Run the scripts from the repo root. Results are written to `results/`.

```
python round1_evolve.py    # ~15-20 min on 2 cores
python round1_analyse.py
python round2_evolve.py    # ~8 min
python round2_analyse.py
python round3_evolve.py    # ~6 min
python round3_analyse.py
```

`results/` already holds the outputs from the original runs. `docs/round1_results_page.html` is the Round 1 results page with interactive charts. Open it in a browser.

## License

MIT. See `LICENSE`.
