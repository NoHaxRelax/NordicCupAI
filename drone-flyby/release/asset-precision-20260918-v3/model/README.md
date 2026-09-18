# Fixed rendered-asset detector bundle

Loads through `drone.scratch_objects.bundle.load_bundle` or the `--bundle` CLI.
All weights were trained from random initialization on mypc. Foreground pixel
matching is deterministic and uses the 241-template bank. Model and template
hashes are checked at load time. The source snapshot is in the parent directory.

See `docs/drone-fixed-asset-result.md` and the parent evaluation receipts for
measured results and data-use limits. This bundle is not an active flight policy.
