# Branch consolidation — 2026-09-07

The user authorized consolidating the repository into one maintained branch: `main`.

## Preserved branch tips

| Former branch | Tip before consolidation | Archival tag |
|---|---|---|
| main | 14216ccc8699839c3a472e32ef95071464732975 | archive/20260907/main-before-unify |
| codex/memory-opt-5090-20260902 | 6e60b7818f7775c2bded7390e6da5df67f338c0e | archive/20260907/memory-opt |
| codex/minimal-ablation-80e-20260903 | bc839dc59cddc8766b8cd05a92854edc00a18339 | archive/20260907/minimal-ablation |
| codex/paper-remaining-evidence-20260905 | 54985930cd3a04fd12a999dc1c35015b9bb281ca | archive/20260907/paper-evidence |

The paper branch contains the latest implementation, experiment records and current plans. Main was fast-forwarded to its tip, then the minimal-ablation branch was merged with a normal two-parent merge commit (`a010f08`). The memory-opt branch was already an ancestor. The minimal-ablation branch had ten additional R3 synchronization commits; its final affected files already matched the paper branch. The merge preserved that history without changing the paper branch's file tree and required no conflict resolution.

## Verification and maintenance

Before publishing, check all four original tips are ancestors of main, inspect tree equality with paper tip before this documentation change, run Python compilation, the release-artifact audit and the CPU MPI policy test. Keep archive tags and a local full-ref bundle (`D:/gpu5090/GraphConditionedPoseMamba_before_unify_20260907.bundle`). Publish main and tags, verify the remote, then remove the three fully merged remote development branches. Do not squash or force-rewrite main.

After consolidation, commit and synchronize future experiment records only to `main`. Historical branch names and source hashes elsewhere in this repository describe provenance, not active destinations. Never recreate those old branches from automation.

The maintained main checkout remains `D:/gpu5090/GraphConditionedPoseMamba-paper-evidence` so current automation paths remain stable. The sibling `D:/gpu5090/GraphConditionedPoseMamba` checkout is retained detached at the consolidated snapshot; it is not a second maintained branch. Existing remote GPU run directories and their frozen sources/checkpoints are not modified by this Git operation.

To inspect a historical state without restoring an old branch, use `git show archive/20260907/paper-evidence:PATH` or create a separate detached worktree at an archive tag. The tags and merge parents preserve every original commit.
