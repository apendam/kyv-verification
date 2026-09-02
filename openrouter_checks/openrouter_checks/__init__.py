"""openrouter_checks: OpenRouter-backed KYV front-image gate sequence + a
standalone, embedding-based duplicate-photo repository.

Two things live here, matching the two scripts this was built for:

  gate_sequence.run_gate_sequence(...)  — walks one upload through the exact
      decision tree in the "KYV Gate Sequence" flowchart (front image -> vehicle
      type -> VRN -> maker -> duplicate -> approve), calling OpenRouter for every
      judgment call and logging tokens/cost/verdict for each step to SQLite.

  duplicate.embed_image / find_best_match / check_duplicate — a from-scratch,
      OpenRouter-embeddings-based near-duplicate detector, independent of the
      SigLIP+pgvector system already in vehicle_front_image_validator/. Backed by
      the same SQLite file (no server, no separate deployment).

Model choice is a parameter everywhere, not a constant — see config.py for the
defaults and how to override them per run.
"""
