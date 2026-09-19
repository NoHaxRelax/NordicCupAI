# Compute status

Our unused GPU pod `zmml46gemukrm5` (`lucas-nikolaj100-20260919`) was deleted
on 2026-09-19. Runpod returned HTTP 204 and it disappeared from inventory.
Its previous 100-game results were already backed up and committed.

Ten running CPU pods are available in the team account, each 32 vCPU and $0.96/h:

| Pod | ID |
|---|---|
| oscar-claude-night-19 | aub4ljf01x7vmr |
| oscar-claude-night-18 | d857rv2nb85pb9 |
| oscar-claude-night-17 | k0yb13rpwnjkkm |
| oscar-claude-night-16 | dcw0waz64nkxqp |
| oscar-claude-night-15 | rdp96oshjmdpot |
| oscar-claude-night-14 | e2p9gq4teyzrtr |
| oscar-claude-night-13 | reud2n6qowb26c |
| oscar-claude-night-12 | xppk5ttbkzx2wf |
| oscar-claude-night-11 | 05g8v2xowqw1v6 |
| oscar-claude-night-10 | u8ycneoqreij5u |

Live CPU utilization was 0% for each. All ten direct SSH probes rejected
`/home/Ucals/.ssh/runpod_codex_team` with `Permission denied (publickey,password)`.
A default-key attempt on night-19 also failed. Their configured startup public
key belongs to Oscar. They have not been restarted, modified, or stopped.
Add the public key in `/home/Ucals/.ssh/runpod_codex_team.pub` to each pod's
`/root/.ssh/authorized_keys` using Oscar's existing access before dispatch.

No Runpod tuning jobs have started. Existing team pods continue billing normally;
this experiment has incurred no new CPU allocation. Remaining historical $10
research allowance is conservatively about $3.94, excluding unrelated team pod
uptime. The planned jobs have a combined active-time ceiling of $3.20.
