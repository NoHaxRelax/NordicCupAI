# Vendored tracker runtime

Copied on 18 September 2026 from the preparation project `drone/perspective_tracking/` (motion model, revisit tracker, placement rules with per-class extents, workflow with the L1 and L2 camera modes and stale-track revisits) with its tests. Runtime needs only NumPy and OpenCV. Do not edit here without porting the change back; regenerate `size-prior.json` with `python -m tracking.placement --fit --annotations src/helsinki/annotations`.

SHA-256 at copy time:

    4d6d36bc8b413e2e03a30861f580f51c7ff3df89afebb8e9d19a7c9424c21113  __init__.py
    e2a3ae6fd2b7bbcb6b8b74ec82297c62d3b527bc37d9461e389c77e54a19070a  motion.py
    47e8dc9b95942829f847826d7323ae9361bbd2a252826c8dbed9b05832bc680d  placement.py
    9b404c570398a742876ca64417de3d7deeece177992dc67e20fa09e8ee44e984  revisit.py
    48dbd71eaefaac087831035420c881df386afb82caa864d8313c3cdb90a5f926  test_placement.py
    01e531fbaf5e48d5eea5b16c79363c0ae3b15493b23d8372ac3a9ed81b63002d  test_revisit.py
    20a685790ea9bf29fc66c1d7bd6dbde098d7b948474f35cb286253d48ac651bf  test_tracker.py
    2c25925349c20816c050beb9397337cc429714e6fee42b19f9b4f6089b720e1f  tracker.py
    708a29fc39f52aec1b2a863c387c6edabcfa12fad47cb1d95f489c9c86b529a7  workflow.py
    c1153a206364eea79aa71e507217c6e35c89bf127b456c1cb68e4e610d69516a  size-prior.json
