# Resource incident — 17 September 2026

Research is paused. Do not clear /tmp/predator-intake-stop or resume automatically.

The user reported an IDE crash. Kernel logs confirmed global out-of-memory
kills of Electron processes at 20:13:43 and 20:15:33 local time. At inspection,
27 GiB RAM was installed and about24 GiB used. Four experiment workers held
approximately15 GiB combined (one6 GiB and three about3 GiB). Replay readers
can additionally deserialize complete recordings. This concurrency was unsafe.

New work was stopped with the shared sentinel. Workers were suspended, then
allowed to finish streaming their current saves one at a time at lower CPU
priority. Three orphaned process-pool workers remained waiting on pipes/locks
after saving; they were terminated after checking they had no open recording
files. The subagent was interrupted. Automatic viewer processing now skips
when the sentinel identifies this IDE incident. Viewer servers were not
restarted, to avoid eager loading of large recordings.

The root full-game attempt bc81e766 saved11747 native frames through1174.6s;
it is a partial run, not a3000s pass. Bait remained alive and one predator was
delivered. Completed earlier recordings and research source files remain on
disk. Do not read all large replays into memory to verify recovery.

Before further simulation: change recording to stream frames to disk, use a
single simulation worker, add an explicit memory ceiling and system-memory
headroom check, and prevent concurrent full-replay deserialization. These
improvements are requirements for resumption, not claims of implemented fixes.
The account-usage stop remains50%, but the incident pause takes precedence.

## Authorized safe resumption

The user subsequently authorized resuming at safe resource use, retaining the
50%-remaining account stop. Native equivalence checks validated disk journals
for frames and then for both frames/events. Tests compare world, every native
frame, events and summary on identical simulation steps.

New run_streaming_v2.py uses debugger/streaming_recorder_v2.py. Jobs run inside
systemd user scopes with MemoryHigh2GiB, MemoryMax3GiB and MemorySwapMax0.
The already-running v1-recorder full-game job was raised to High3GiB/Max4GiB
while system memory headroom exceeded13GiB; it still retains its original
in-memory event list. Future jobs journal events too. A resource watchdog
stops all experiment loops gracefully below6GiB available RAM or20GiB disk
free, or at50% account usage remaining. Each research batch runs sequentially;
independent capped batches can run concurrently while headroom permits.

Viewer frame generation now iterates original recordings and writes100-frame
chunks without deserializing the entire recording. The catalog has an explicit
--verify-stream mode for bounded-memory discovery/validation. The original
catalog server remains stopped because its default scan and full-replay browser
loading still allocate whole recordings. The lightweight9055 viewer is running.
