import json

from fdb.core.models import FailureType


def test_failure_codes_remain_strings_on_python310():
    failure = FailureType.ALIGNMENT_FAILURE
    assert str(failure) == "ALIGNMENT_FAILURE"
    assert f"{failure}" == "ALIGNMENT_FAILURE"
    assert json.loads(json.dumps({"failure": failure})) == {"failure": "ALIGNMENT_FAILURE"}
    assert FailureType("ALIGNMENT_FAILURE") is failure
