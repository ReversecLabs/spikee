def attack(
    entry,
    target_module,
    call_judge,
    attack_iterations,
    attempts_bar,
    global_lock,
    attack_options=None,
):
    last_input = ""
    last_response = ""
    success = False

    for iteration in range(attack_iterations):
        candidate_input = f"[mock-attack-{iteration}] {entry['text']}"
        last_input = candidate_input
        response, _ = target_module.process_input(
            candidate_input, entry.get("system_message")
        )
        last_response = response
        success = call_judge(entry, response)
        with global_lock:
            attempts_bar.update(1)
        if success:
            return iteration + 1, True, last_input, last_response

    return attack_iterations, False, last_input, last_response
