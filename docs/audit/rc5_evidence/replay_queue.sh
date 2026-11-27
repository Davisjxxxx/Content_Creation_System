#!/usr/bin/env bash
# Run the desktop recovery fixture with a disposable network, HOME and process namespace.
set -euo pipefail
evidence=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd "$evidence/../../.." && pwd)
bundle=$(realpath "${1:?Supply the verified RC5 bundle directory}")
venv=$(realpath "${2:?Supply the existing matching test venv directory}")
test -x "$bundle/AvatarV2"
test -x "$venv/bin/python"
scratch=$(mktemp -d)
mkdir -m 700 "$scratch/home" "$scratch/tmp" "$scratch/runtime"
echo "Fixture receipts in $scratch"
bwrap --die-with-parent --unshare-all --new-session --clearenv \
    --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
    --symlink usr/sbin /sbin --ro-bind /etc /etc --dev /dev --proc /proc --ro-bind /sys /sys \
    --tmpfs /run --bind "$scratch/tmp" /tmp --bind "$scratch/home" /home/test --bind "$scratch/runtime" /runtime \
    --ro-bind "$repo/tests/fixtures/rc5/telemetry-bin" /fixture-telemetry --ro-bind "$evidence" /checks --ro-bind "$bundle" /rc --ro-bind "$venv" /venv \
    --ro-bind "$repo/scripts/launch_avatar_v2.sh" /src/scripts/launch_avatar_v2.sh \
    --chdir /tmp --setenv PATH /fixture-telemetry:/venv/bin:/usr/bin:/bin --setenv HOME /home/test \
    --setenv XDG_RUNTIME_DIR /runtime --setenv AVATAR_V2_HOME /home/test/app \
    --setenv LANG C.UTF-8 --setenv PYTHONDONTWRITEBYTECODE 1 python /checks/queue_restart_probe.py
