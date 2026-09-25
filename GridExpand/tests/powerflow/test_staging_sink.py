"""DbRunSink writes to a staging run and swaps it in only when Step 4 completes."""

from gridexpand.db.runs import STAGING_MARKER, staging_run_name
from gridexpand.powerflow.io import DbRunSink


class FakeDb:
    def __init__(self):
        self.calls = []

    def create_powerflow_run(self, grid_ref, **kwargs):
        self.calls.append(("create", kwargs["run_name"]))
        return 41

    def promote_powerflow_run(self, run_id, run_name):
        self.calls.append(("promote", run_id, run_name))

    def discard_powerflow_run(self, run_id):
        self.calls.append(("discard", run_id))


def _sink(db):
    return DbRunSink(
        db, {"grid_result_id": 1}, urbs_input_file="x.h5", pre_only=True,
        scenario_key="s", run_name="final_raw_powerflow", assumptions={},
    )


def test_staging_name_is_unique_and_marked():
    assert staging_run_name("a", "t1").startswith(f"a{STAGING_MARKER}")
    first, second = FakeDb(), FakeDb()
    _sink(first), _sink(second)
    name_a, name_b = first.calls[0][1], second.calls[0][1]
    assert name_a != name_b
    assert name_a.startswith(f"final_raw_powerflow{STAGING_MARKER}")


def test_promote_and_discard_use_the_staging_run():
    db = FakeDb()
    sink = _sink(db)
    sink.promote()
    sink.discard()
    assert db.calls[1:] == [("promote", 41, "final_raw_powerflow"), ("discard", 41)]
