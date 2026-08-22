# ClothDojo agent instructions

ClothDojo is the clean successor to `newton-cloth-rl`. The old repository is a
read-only source of behavior and evidence, not a package dependency.

## Product boundary

- The long-term product is a reproducible cloth-manipulation benchmark.
- Teleoperation, π0.5 training, STEAM and rollout form the data-agent bootstrap.
- The trained agent is an input to benchmark generation, not part of the
  benchmark definition.

## Simplicity constraints

- Do not add a `patches/`, `vendor/`, or catch-all `data/runtime/` directory.
- Keep one project environment defined by `requirements.txt`; do not add a
  generated dependency lock without a demonstrated reproducibility failure.
- Use released Newton 1.5 APIs directly. Do not add Newton 1.4 compatibility
  branches.
- Treat RLINF as a pinned upstream dependency behind a small adapter. Before
  proposing a fork, compatibility layer, or replacement, verify the latest
  upstream source and prove the remaining gap with a minimal diff and test.
  Fork only when the required change cannot live in ClothDojo or be upstreamed;
  never apply patches during setup.
- Keep datasets, checkpoints, caches and generated outputs outside the Git
  repository. Configurations reference them explicitly.
- Do not copy old scripts in bulk. Migrate one user-visible workflow at a time
  with an acceptance test.
- Distinguish "missing from the currently pinned version" from "missing
  upstream" before making an architectural decision.

## Documentation

- `docs/project_index.md` locates stable project entry points.
- `docs/architecture.md` owns system boundaries and dependency direction.
- `docs/migration.md` tracks behavior migrated from `newton-cloth-rl`.
- `docs/todo.md` records deferred decisions with meaningful long-term risk.
