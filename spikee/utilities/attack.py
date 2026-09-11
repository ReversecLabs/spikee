"""Compatible attack invocation and progress accounting."""

import inspect
import os


def attack_history_enabled():
    """Whether supporting attacks should collect optional diagnostic history."""
    return os.getenv("SPIKEE_ATTACK_HISTORY", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def invoke_attack(
    attack,
    entry,
    target,
    judge,
    iterations,
    bar,
    lock,
    options,
):
    """Old modules receive exactly the optional arguments they declare."""
    parameters = inspect.signature(attack).parameters
    kwargs = {}
    for name in ("attack_options", "attack_option"):
        if name in parameters:
            kwargs[name] = options
            break
    return attack(entry, target, judge, iterations, bar, lock, **kwargs)


class AttackProgress:
    """Isolate an attack's budget adjustments from the shared progress bar."""

    def __init__(self, bar, iterations):
        self.bar = bar
        self.total = iterations
        self.n = 0

    def update(self, count=1):
        self.n += count
        self.bar.update(count)

    def refresh(self):
        self.bar.refresh()
