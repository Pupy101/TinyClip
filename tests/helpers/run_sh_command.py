from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: List[str]) -> None:
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as exc:
        msg = exc.stderr.decode()
    if msg:
        pytest.fail(msg=msg)
