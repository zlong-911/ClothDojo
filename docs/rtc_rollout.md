# RTC rollout

RTC rollout is an offline data-production workflow. Simulation intentionally
treats policy inference as zero-latency so wall-clock server speed cannot alter
the generated trajectory.

## Control loop

Both execution modes use the same synchronous cadence:

```text
chunk: observe -> predict 50 actions -> execute N actions -> replan

RTC:   observe -> guided prediction using the old tail
       -> execute N actions -> replan with delay=0
```

- `chunk` predicts every chunk independently.
- `rtc` passes the unexecuted model-space tail into the next denoising call.
  Guidance softly aligns the new prefix with that tail before the new chunk
  takes over.
- `--exec-horizon` controls the cadence for both modes and defaults to 10.
- Inference delay is always zero for offline production. Real-time delay belongs
  to a separate evaluation concern and is not mixed into the dataset generator.

The default production mode is the synchronous chunk baseline:

```bash
clothdojo rollout run \
  --scenario configs/rollout/shirt_flatten.yaml \
  --states data/states \
  --policy ws://127.0.0.1:8080 \
  --seen-assets 5 \
  --unseen-assets 5 \
  --limit-per-asset 2 \
  --gpus 0 \
  --output data/rollouts/sft_chunk
```

It predicts 50 actions, executes the first 10, then observes and replans. Add
`--execution rtc` to keep the same cadence and enable guided overlap.

Each active simulation worker requires a unique policy endpoint. For example,
two workers use `--gpus 0,1 --policy ws://policy-a:8080,ws://policy-b:8080`.
Sharing one policy process would interleave its global sampling RNG across
episodes, so the coordinator rejects that topology before launching workers.

## Policy boundary

The SFT service uses the pinned RLINF OpenPI model and its existing input/output
transforms. Guided RTC is a small ClothDojo adapter at the flow-ODE sampling
boundary: it matches LeRobot's released guided correction with a linear prefix,
execution horizon 10 and maximum weight 10. It does not patch RLINF or upgrade
the pinned LeRobot data dependency. Previous delta actions are rebased from
their original observation state to the current state before guidance. The
client records the checkpoint, normalization and RTC identities automatically.
The policy service exposes `--rtc-prefix-schedule`,
`--rtc-max-guidance-weight` and `--rtc-execution-horizon` for controlled RTC
experiments; their defaults remain `linear`, 10 and 10.

## Data flow

```text
data/states/<asset>/states.npz
  -> data/rollouts/<run>/trajectories
  -> data/rollouts/<run>/lerobot
```

The trajectory is the canonical artifact and uses the same physical format as
teleoperation. The LeRobot dataset is rendered later by exact-state replay.
Production metadata records the execution mode, zero-delay contract, scenario,
policy identity, initial-state identity, and NOCS visual contract.

Each GPU worker claims whole asset jobs. Existing compatible trajectories are
reused when the same command resumes; workers never write root metadata
concurrently.
