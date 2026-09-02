"""openrouter_checks: OpenRouter-backed KYV front-image gate sequence + a
standalone, local (no-API) duplicate-photo repository.

Two things live here, matching the two scripts this was built for:

  gate_sequence.run_gate_sequence(...)  — walks one upload through the exact
      decision tree in the "KYV Gate Sequence" flowchart (front image -> vehicle
      type -> VRN -> maker -> duplicate -> approve), calling OpenRouter for every
      judgment call and logging tokens/cost/verdict for each step to SQLite.

  duplicate.check_duplicate(...) — near-duplicate detection over two local
      signals, cheapest first: pHash (instant, catches near-identical
      reuploads), falling back to a local SigLIP vector embedding (same
      weights vehicle_front_image_validator/ uses for its own duplicate
      check) only when pHash comes back clean. Neither makes a network
      call. Backed by the same SQLite file (no server, no separate
      deployment).

Model choice is a parameter everywhere, not a constant — see config.py for the
defaults and how to override them per run.
"""
