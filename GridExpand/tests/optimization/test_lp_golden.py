"""The solver LP file of a tiny synthetic input is pinned (see lp_fixture.py).

The hashes were computed with the model code before the 2026-09 cleanup
(commit cc895ea) and are independent of PYTHONHASHSEED. A change here means
the heuristic LP can return another optimal vertex: verify with the harness.
"""

from lp_fixture import lp_sha256

ONE_CLUSTER = ["6a9ad20221df8afdb1954cbfe121ecf319302bcca9919a40f36374d275aeacfe"]
TWO_CLUSTERS = [
    "57b7f7f184af613353bcfbda38be4ca2954edc237ffaea9c0848c7b3f27cba2b",
    "5e3fbe682584aba91542abbc5a350482deda36c7c5b0e369e41b3b0208b20812",
]


def test_lp_file_is_unchanged(tmp_path):
    assert lp_sha256(tmp_path, 1) == ONE_CLUSTER
    assert lp_sha256(tmp_path, 2) == TWO_CLUSTERS
