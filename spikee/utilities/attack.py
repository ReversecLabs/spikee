"""Compatible attack invocation and progress accounting."""

import inspect


def accepts_attack_history(attack):
    return "return_all_attempts" in inspect.signature(attack).parameters


def invoke_attack(
    attack,
    entry,
    target,
    judge,
    iterations,
    bar,
    lock,
    options,
    return_all_attempts=False,
):
    """Old modules receive exactly the optional arguments they declare."""
    parameters = inspect.signature(attack).parameters
    kwargs = {}
    for name in ("attack_options", "attack_option"):
        if name in parameters:
            kwargs[name] = options
            break
    if accepts_attack_history(attack):
        kwargs["return_all_attempts"] = return_all_attempts
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
