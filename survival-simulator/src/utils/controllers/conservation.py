"""Gradually spend less energy on optional scouting after the shared map exists.

The caller decides which tasks are optional. Known meals, escape and initial
mapping should never be held back by this schedule. Only public simulation
time and agent IDs are used here.
"""

from dataclasses import dataclass
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConservationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    enabled: bool = Field(default=True, strict=True)
    start_seconds: float = Field(default=300., ge=0)
    full_seconds: float = Field(default=1800., gt=0)
    late_scouting_fraction: float = Field(default=.3, ge=0, le=1)
    late_scan_interval_seconds: float = Field(default=18., ge=6)
    scouting_cycle_seconds: float = Field(default=10., gt=0)

    @model_validator(mode="after")
    def valid_times(self):
        if self.full_seconds <= self.start_seconds:
            raise ValueError("full_seconds must be greater than start_seconds")
        return self


@dataclass
class _Scan:
    next_scan: float
    remaining_ticks: int = 0
    last_call: float | None = None
    last_dt: float = .1
    last_result: bool = False


class ConservationSchedule:
    EARLY_SCAN_SECONDS = 6.
    SCAN_TICKS = 8
    AGENT_PHASE_SECONDS = .37

    def __init__(self, config: ConservationConfig | None = None):
        self.config = config if config is not None else ConservationConfig()
        self.reset()

    def reset(self):
        self.scans: dict[int, _Scan] = {}

    def prune(self, living_ids):
        living = set(living_ids)
        self.scans = {key: scan for key, scan in self.scans.items() if key in living}

    def fraction(self, now):
        """Continuous transition from early exploration (0) to conservation (1)."""
        if not self.config.enabled:
            return 0.
        return min(1., max(0., (now - self.config.start_seconds)
                           / (self.config.full_seconds - self.config.start_seconds)))

    def scouting_fraction(self, now):
        fraction = self.fraction(now)
        return (1. - fraction) + fraction * self.config.late_scouting_fraction

    def scan_interval(self, now):
        return self.EARLY_SCAN_SECONDS + self.fraction(now) * (
            self.config.late_scan_interval_seconds - self.EARLY_SCAN_SECONDS)

    def should_scout(self, agent_id, now):
        duty = self.scouting_fraction(now)
        if duty >= 1.:
            return True
        cycle = self.config.scouting_cycle_seconds
        phase = (now + agent_id * self.AGENT_PHASE_SECONDS) % cycle
        return phase < duty * cycle

    def scanning(self, agent_id, now, dt):
        """Whether to issue this tick's 45-degree stationary scan.

        Deadlines are fixed when scheduled, so the changing interval cannot
        shift a sweep halfway through. Each sweep has eight consecutive
        ticks. If another task interrupts it, the next eligible call starts
        a fresh complete sweep. Repeated calls for one tick are idempotent.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        if not self.config.enabled:
            # Exact historical scan timing makes disabled runs a useful
            # control, without changing their turn commands or RNG use.
            return ((now + agent_id * self.AGENT_PHASE_SECONDS)
                    % self.EARLY_SCAN_SECONDS < self.SCAN_TICKS * dt)

        scan = self.scans.get(agent_id)
        if scan is None:
            interval = self.scan_interval(now)
            phase = (now + agent_id * self.AGENT_PHASE_SECONDS) % interval
            offset = (-now - agent_id * self.AGENT_PHASE_SECONDS) % interval
            # Joining an already active scan window should not leave a new
            # stationary agent blind until the following cycle. Give it a
            # complete sweep, retaining a staggered next deadline.
            scan = self.scans[agent_id] = (_Scan(now + interval + phase, self.SCAN_TICKS)
                if phase < self.SCAN_TICKS * dt else _Scan(now + offset))
        if scan.last_call == now:
            return scan.last_result

        interrupted = (scan.remaining_ticks > 0 and scan.last_call is not None
                       and now - scan.last_call > 1.5 * max(dt, scan.last_dt))
        if interrupted or (scan.remaining_ticks == 0 and now >= scan.next_scan - 1e-9):
            scan.remaining_ticks = self.SCAN_TICKS
            scan.next_scan = now + self.scan_interval(now)
        result = scan.remaining_ticks > 0
        if result:
            scan.remaining_ticks -= 1
        scan.last_call, scan.last_dt, scan.last_result = now, dt, result
        return result
