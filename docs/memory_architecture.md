# Memory architecture

## Design objective

FDB records broadly but recalls selectively. The repository is an external long-term
memory, not a prompt-sized working memory. Records must contain reusable evidence and
decision summaries, never hidden chain-of-thought.

## Hierarchy

```text
Raw observations and execution output
  -> episodes (immutable evidence)
    -> summaries (compressed, source-linked history)
      -> long-term memory (validated general rules)
        -> current memory (small task-specific retrieval set)
```

Skills and robot self-models sit beside this hierarchy. They cite the episodes,
experiments, research logs, and source material that justify them.

## Record flow

1. **Observe:** capture confirmed world/robot state and its provenance.
2. **Decide:** store the interpreted goal, candidates, chosen plan, and a concise
   selection reason.
3. **Simulate and act:** store predicted metrics separately from actual metrics.
4. **Evaluate:** record objective results first; attach user and critic evaluations as
   distinct fields.
5. **Summarize:** compress related episodes while retaining their IDs and exceptions.
6. **Promote:** create a long-term rule or skill only after the policy gates pass.
7. **Retrieve:** load only records relevant to the active goal, morphology, environment,
   and failure mode.
8. **Correct:** mark contradicted knowledge deprecated, link replacement evidence, and
   write an audit event.

## Operations

| Operation | Contract |
| --- | --- |
| `read_memory` | Read an exact record by stable ID or path. |
| `search_memory` | Search metadata and summaries before loading raw history. |
| `write_memory` | Create a new record without overwriting evidence. |
| `update_memory` | Create a revision and an audit event; retain provenance. |
| `archive_memory` | Move inactive records out of normal retrieval, preserving them. |
| `promote_memory` | Convert repeated evidence into a scoped long-term rule. |
| `deprecate_memory` | Mark unsafe or contradicted knowledge and link its replacement. |
| `create_skill` | Register only after reproducible simulation evidence and review gates. |
| `update_skill` | Version behavior, prerequisites, limits, tests, and evidence together. |
| `search_old_episode` | Query episode metadata, summaries, outcome, robot, and tags. |
| `summarize_history` | Produce a source-linked synthesis without destroying raw records. |

## Retrieval order

FDB should query current memory, relevant skills, the matching robot model, and
long-term rules first. It then searches summaries, raw episodes, local research, and
finally external sources. Retrieval results are hypotheses until checked against the
current state and applicable constraints.

## Objective metrics and critics

Task completion, final position, collisions, drops, joint-limit violations, execution
time, and path length are objective observations. Strategy quality, smoothness, grasp
appropriateness, process quality, and predicted user preference are critic outputs.
The two groups are stored separately so a persuasive critic cannot overwrite physical
failure.

## Git history

Commits are historical checkpoints, not the only data index. Records use stable IDs,
timestamps, schema versions, status, and evidence links so they remain searchable
without replaying the entire Git history. Material changes and all deprecations require
an audit record and a focused commit.

