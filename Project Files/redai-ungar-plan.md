
Make **UNGAR** completely RediAI-agnostic, then build a **thin bridge package** that knows about _both_ — so UNGAR never imports RediAI, but RediAI (optionally) imports UNGAR and plugs in via its existing registry / XAI / tournament / workflow systems.

That gives you:

* ✅ No hard dependency on RediAI

* ✅ No future “big refactor to make them play nice”

* ✅ Ability to turn on “full RediAI superpowers” when you want

Let’s make that concrete.

* * *

1. High-level architecture: “core + bridge” model

-------------------------------------------------

Think of three layers:

1. **UNGAR Core (this project)** – your card physics + 4×13×n substrate:
   
   * Card/zone/phase engine
   
   * 4×13×n tensor encoder
   
   * Generic `Env` + `Agent` interfaces
   
   * Simple local training loop(s)
   
   * Tiny metrics/logging abstraction

2. **Bridge Package (separate repo / optional pip extra)** – e.g. `ungar-rediai-bridge`:
   
   * Imports **UNGAR** and **RediAI**
   
   * Implements RediAI’s **environment adapter** around UNGAR’s envs
   
   * Uses RediAI’s **WorkflowRecorder / registry / RewardLab / XAI hooks** where available
   
   * Ships example workflows (`.yaml`) that run UNGAR training inside RediAI

3. **RediAI Platform (unchanged)** – workflow registry, XAI suite, RewardLab, tournaments, etc., as already specced.

**Key rule:**

> UNGAR core does _not_ import or mention RediAI. RediAI (or the bridge) depends on UNGAR, never the other way around.

* * *

2. What UNGAR needs to expose so you _never_ have to refactor for RediAI

------------------------------------------------------------------------

Design a few **stable, generic contracts** now. They don’t say “RediAI” anywhere, but they are _exactly_ what the bridge needs later.

### 2.1 Environment interface

Something like:
    class UngarEnv(Protocol):
        def reset(self, seed: int | None = None) -> Obs:
            ...

        def step(self, action: Action) -> tuple[Obs, float, bool, dict]:
            ...

        def legal_actions(self) -> list[Action]:
            ...

        def current_player(self) -> int: ...

Where `Obs` **always** contains (or can produce):

* The **4×13×n tensor** (numpy / torch)

* Minimal scalar context: phase id, pot/score, trick index, etc.

This lines up perfectly with RediAI’s existing “game env adapter” concept and transformer state encoders.

### 2.2 Agent interface

A game-agnostic `Agent`:
    class Agent(Protocol):
        def act(self, obs: Obs) -> Action: ...

The bridge will wrap RediAI policies (FiLM actor, transformer, etc.) to implement this.

### 2.3 Telemetry / hooks interface

Define a _tiny_ callback protocol in UNGAR like:
    class TrainingListener(Protocol):
        def on_episode_start(self, meta: dict): ...
        def on_step(self, step: dict): ...
        def on_episode_end(self, summary: dict): ...

* Core UNGAR ships a **NoOpListener** and maybe a simple **CSV/JSON logger**.

* The bridge package will implement a **RediAITrainingListener** that forwards these to:
  
  * `WorkflowRecorder` for metrics & artifacts
  
  * RewardLab (`reward_decompositions` table)
  
  * XAI/overlay storage (`xai_analyses`, `peeker_models`, etc.)

This is how you avoid future surgery: all your runs already “speak” a simple struct; RediAI just listens.

### 2.4 XAI / overlay contract

Given your XAI-first goals in UNGAR’s vision (heatmaps on the 4×13 grid, attribution per decision), agree on a **canonical overlay format** now:
    @dataclass
    class CardOverlay:
        importance: np.ndarray  # shape (4, 13)
        label: str              # e.g. "policy_logit_gradients"
        meta: dict              # method, seed, etc.

UNGAR doesn’t need Captum/SHAP/etc.; RediAI already has those.  
But if UNGAR can _accept or emit_ `CardOverlay` objects, the bridge can:

* Run RediAI’s XAI methods over UNGAR decisions

* Store them in RediAI’s XAI tables

* Render 4×13 overlays in either UI

Again: no refactor later because the data shape is already agreed upon.

* * *

3. How to wire RediAI in without entangling it

----------------------------------------------

### 3.1 Bridge package responsibilities

`ungar-rediai-bridge` (name flexible) will:

1. **Env adapter**  
   Map UNGAR env → RediAI’s `EnvAdapter` / serving interface:
   
   * `step/reset` → RediAI match runner
   
   * 4×13×n tensor → RediAI `GameStateEncoder` input
   
   * Per-player obs → RediAI personality / FiLM conditioning

2. **Workflow integration**  
   Provide helper functions:
      def run_ungar_training_with_rediai(env_name: str, workflow_id: str | None = None): ...

  Internally:

* Creates / attaches to a Workflow Registry entry

* Uses `WorkflowRecorder` to log metrics, checkpoints, overlays, etc.
3. **Reward and tournament mapping**
   
   * Route UNGAR reward components into **RewardLab** (reward decomposition tables).
   
   * Hook UNGAR bots into RediAI **tournament** & **agent_ratings** tables.

4. **Example workflows**
   
   * Ship `examples/workflows/ungar_hearts_training.yaml`, etc., that RediAI can run out-of-the-box.

### 3.2 Dependency shape

* `ungar` – **no** RediAI in `pyproject.toml`

* `rediai` – unchanged

* `ungar-rediai-bridge` – **depends on both**:
  
      [project.optional-dependencies]
      rediai = ["rediai>=2.1", "ungar>=0.1"]
  
  

If you want, UNGAR itself can expose:
    [project.optional-dependencies]
    rediai = ["ungar-rediai-bridge"]

So users can `pip install ungar[rediai]` and get the glue automatically, but the core library remains clean.

* * *

4. Phased plan with small, end-to-end milestones

------------------------------------------------

Keeping with your “small phases, always tested E2E” preference:

### Phase 0 – Lock UNGAR’s public contracts (no RediAI involved)

**Milestone 0.1: Minimal core API**

* Implement:
  
  * `UngarEnv` protocol
  
  * `Agent` protocol
  
  * 4×13×n encoder

* Ship one tiny game (e.g., “High Card Duel”) with a random agent.

* Test: run 100 episodes and confirm the tensor API + invariants.

**Milestone 0.2: Telemetry hooks**

* Add `TrainingListener` interface and plug it into your simple trainer.

* Implement `NoOpListener` + `JSONLinesListener`.

* Test: run a toy training loop and verify logs (no RediAI yet).

* * *

### Phase 1 – Local training & overlays inside UNGAR

**Milestone 1.1: Simple local trainer**

* Implement a small, framework-free trainer (e.g., tabular Q or policy gradient using PyTorch).

* Use the 4×13×n obs and TrainingListener.

* Test: run on one simple game to completion.

**Milestone 1.2: Local XAI scaffold**

* Define the `CardOverlay` dataclass.

* Provide a _dummy_ attribution method (e.g., random or simple heuristics) just to exercise the path.

* Extend `TrainingListener` or a sibling to handle overlays per step/episode.

* Test E2E: from game → decision → overlay object → serialized to JSON.

Now UNGAR is fully self-sufficient and already shaped for RediAI.

* * *

### Phase 2 – Build the bridge (in a separate repo)

**Milestone 2.1: Basic env adapter**

* Create `ungar-rediai-bridge` repo.

* Implement a RediAI env adapter that:
  
  * Wraps a UNGAR game env
  
  * Produces the correct `GameStateEncoder` inputs and metadata

* Test: RediAI can step the UNGAR env through a full episode using a trivial policy.

**Milestone 2.2: WorkflowRecorder integration**

* Implement a `RediAITrainingListener` that:
  
  * On `on_step` / `on_episode_end`, calls `WorkflowRecorder.record_metric` / `record_artifact`.

* Provide a helper:
  
      async def train_with_rediai(env: UngarEnv, workflow_id: str | None = None): ...

* Test E2E: run training through RediAI; confirm metrics appear in `workflows`, `workflow_events`, etc.

**Milestone 2.3: XAI + RewardLab mapping**

* Implement functions to:
  
  * Turn RediAI XAI outputs into `CardOverlay` and vice versa.
  
  * Log RewardLab components for UNGAR rewards into `reward_decompositions`.

* Test: one UNGAR run produces:
  
  * At least one XAI analysis row
  
  * At least one reward decomposition row.

* * *

### Phase 3 – Guardrails and quality

Since you care about guardrails and enhancements when wiring workflows:

1. **Versioning guardrails**
   
   * Give UNGAR a semantic version and **document** the stable interfaces (`UngarEnv`, tensor layout, TrainingListener, CardOverlay).
   
   * In the bridge, assert compatible versions at import time and fail loudly with a helpful message.

2. **Feature flags / graceful degradation**
   
   * In the bridge, if `WorkflowRecorder` / XAI / RewardLab modules are missing or misconfigured, fall back to:
     
     * Local logging only
     
     * No XAI / metrics, but **never** break the training run.

3. **Testing guardrails**
   
   * Add E2E tests in the bridge that:
     
     * Spin up a minimal RediAI in test mode
     
     * Run a UNGAR game through a workflow
     
     * Assert a small set of DB rows exists (`workflows`, `gt_metrics`, `xai_analyses`, etc.).

4. **CI guardrails**
   
   * Bridge repo CI:
     
     * Runs UNGAR unit tests (or at least a smoke test) against the pinned version
     
     * Runs RediAI’s `run_tests.py --unit` in a light mode (or a focused subset) to catch API breaks.


