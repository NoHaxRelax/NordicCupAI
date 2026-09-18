# Validation

- Included source bytes match the SHA-256 manifest; replay files were removed before publication.
- All 18 vendored simulator blobs match the recorded upstream Git source hashes.
- Python sources parse successfully.
- `verify_findings.py` passes checks of the headline wall-renewal, wall-availability, delivery/bodyguard, gap-availability, native holds and failure-control counts against retained JSON.
- The wall v7 controller matches archived hash `484f61dfee082c05c5673ea1646b129cbd3d2614a35837e695c79aa30cc584e1`.
- All branch changes are additions inside `survival-simulator/oscar-trap-research/`; existing team code is unchanged.
- Replay data, large-file chunks and the embedded offline viewer are excluded. Historical reports retain references to local recordings as provenance.

No new simulation or competition evaluation was run for this handoff. Experimental limitations remain documented in the reports.
