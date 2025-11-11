# Installing Spikee in an Isolated Environment (zip upload only)

**Scenario:** you have a completely isolated Linux system (e.g. Kali) in a client environment that you can use for testing an LLM application but this system has **no internet access and no pip**. The only thing you can do is **upload a single ZIP/TAR file** and extract it before the test, and at the end of the test download a ZIP file with the results.

 The goal is to build Spikee on your local machine, package it with all dependencies, and then run it on the remote system without any installation.

---

## Approach

We build Spikee in a local Python virtual environment (`venv`), pack that environment into a single archive, upload it, extract it on the target, and run Spikee directly from it. This way, you run Spikee on a system with no pip and no internet.

---

## Requirements

Before building locally, make sure your build system and the remote system are compatible:

* Both must use the **same CPU architecture** (e.g. x86_64 ↔ x86_64).
* Both must use the **same Python version** (e.g. 3.11.x on both).
* The remote must already include essential system libraries (libssl, libpcap, etc.).
* Build on a host that has a **similar or older glibc version** than the remote to avoid binary incompatibilities.

---

## Matching the remote Python version and architecture

Ask the client to provide this information from the remote system:

```bash
python3 --version
uname -m
```

Once you know these, ensure your local build host matches them. If your Python version differs, install the same one locally:

```bash
# Example: install Python 3.11 on Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-distutils

# Ensure it’s the one you’ll use to build the package
python3.11 --version
```

You can then use that version explicitly in all commands, e.g.:

```bash
python3.11 -m venv env_spikee
```

If your distro doesn’t offer that Python version, you can install it quickly via [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) or use tools like `pyenv`.

---

## Steps on your local machine

```bash
# 1. Verify environment
python3 --version   # must match remote Python version
uname -m            # must match remote architecture

# 2. Create and prepare a virtual environment
python3 -m venv env_spikee
source env_spikee/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 3. Install Spikee and its dependencies
pip install . --prefer-binary 

# 4. Verify it works
env_spikee/bin/spikee

# 5. Clean up and package for transfer
rm -rf env_spikee/.cache
find env_spikee -name '__pycache__' -exec rm -rf {} +
tar -czf spikee_env.tar.gz env_spikee
```

---

## Steps on the remote (isolated) system

Upload `spikee_env.tar.gz` by whatever method is available (e.g. USB, file upload).

```bash
# 1. Extract to /tmp or your working directory
tar -xzf spikee_env.tar.gz -C /tmp

# 2. Work interactively
source /tmp/env_spikee/bin/activate
spikee list seeds
deactivate
```

---

## Troubleshooting

* **“Error importing numpy: you should not try to import numpy from its source directory”** → Rebuild on a matching host and use `--prefer-binary` to ensure binary wheels.
* **“Wrong ELF class” or “undefined symbol”** → Architecture or libc mismatch. Rebuild on a host identical to the remote.
* **Missing system libs** → Install them on the remote before running Spikee.

---

## Summary

You’re building Spikee locally and shipping the entire Python environment as a self-contained archive. The remote system only needs Python preinstalled — no pip, no internet, no installation.
