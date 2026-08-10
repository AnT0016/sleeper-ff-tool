"""Offline unit tests for the collection crons (``.github/workflows/collect-*.yml``).

The two capture workflows are the only part of the pipeline that is *configuration* rather than
code, and they carry decisions the Python test suite cannot see. Each assertion below stands for a
silent failure that was designed out of the system, and that a plausible future edit would design
back in:

* **Nothing pinned on the scheduled path.** A hardcoded ``--week`` would fix every Tuesday to one
  number forever and undo #16's schedule-based week resolution. It is exactly the change someone
  debugging a flaky cron would reach for.
* **``LAKE_BACKEND=s3``, and a guard that refuses anything else.** An absent value resolves to the
  local-parquet dev backend (``store.lake``), which writes to the runner's disk and is destroyed
  with it -- a green run that captures nothing.
* **No ``continue-on-error`` / ``|| true``.** ``collect.py`` exits 1 when a forward-only source
  fails because those rows are unrecoverable; swallowing that exit code is what makes the loss
  silent (#14's review).
* **One shared concurrency group across BOTH files.** ``store.write_snapshot`` is read-modify-write
  on a partition, so mutual exclusion is what stops a manual dispatch from losing an update to a
  scheduled run. It holds only because the two files spell the same literal string -- a cross-file
  coincidence that no single-file review would catch breaking.
* **No git-commit step.** Data goes to the bucket; a commit step would be a rollback to the
  pre-#9 design.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

#: The two capture crons, and the cadence each one runs.
CAPTURE: dict[str, str] = {
    "collect-prelock.yml": "prelock",
    "collect-postgame.yml": "postgame",
}

#: The four ``LAKE_S3_*`` values ``store.s3`` requires. ``LAKE_S3_REGION`` is deliberately absent --
#: it is optional and derived from the endpoint host.
REQUIRED_SECRETS = (
    "LAKE_S3_ENDPOINT",
    "LAKE_S3_ACCESS_KEY_ID",
    "LAKE_S3_SECRET_ACCESS_KEY",
    "LAKE_S3_BUCKET",
)


def _load(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _on(workflow: dict) -> dict:
    """The ``on:`` block. YAML 1.1 reads a bare ``on`` as the boolean ``True``, not the string."""
    return workflow.get("on") if "on" in workflow else workflow[True]


def _job(workflow: dict) -> dict:
    jobs = list(workflow["jobs"].values())
    assert len(jobs) == 1, "one job per capture workflow keeps the concurrency guard meaningful"
    return jobs[0]


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _run_steps(job: dict) -> list[str]:
    return [step["run"] for step in job["steps"] if "run" in step]


# --------------------------------------------------------------------------- the two files exist
def test_both_capture_workflows_are_present_and_parse():
    for name in CAPTURE:
        assert (WORKFLOWS / name).is_file(), f"{name} is missing"
        assert _load(name), f"{name} parsed as empty"


@pytest.mark.parametrize("name", CAPTURE)
def test_each_workflow_can_be_run_by_hand(name):
    """``workflow_dispatch`` is the recovery path -- #16's stepped-over-week WARNING names it."""
    assert "workflow_dispatch" in _on(_load(name))


def test_the_prelock_crons_cover_thursday_and_sunday():
    crons = [entry["cron"] for entry in _on(_load("collect-prelock.yml"))["schedule"]]
    assert len(crons) == 2
    hours_and_days = {(c.split()[1], c.split()[4]) for c in crons}
    assert hours_and_days == {("22", "4"), ("15", "0")}


def test_the_postgame_cron_is_tuesday_before_the_season_db_refresh():
    crons = [entry["cron"] for entry in _on(_load("collect-postgame.yml"))["schedule"]]
    assert len(crons) == 1
    _, hour, _, _, day = crons[0].split()
    assert (hour, day) == ("12", "2")
    # refresh.yml rebuilds season.db at 18:00 UTC Tuesday; the lake capture must land before it.
    refresh = [entry["cron"] for entry in _on(_load("refresh.yml"))["schedule"]]
    assert int(hour) < int(refresh[0].split()[1])


# --------------------------------------------------------------------------- the scheduled path
@pytest.mark.parametrize(("name", "mode"), CAPTURE.items())
def test_the_scheduled_path_pins_neither_season_nor_week(name, mode):
    """#16 resolves the postgame week from the schedule; a literal ``--week`` here would undo it.

    Season and week may only reach the script through the ``workflow_dispatch`` inputs, which are
    empty on every scheduled run. A hardcoded value would pin every fire to one number forever.
    """
    run = "\n".join(_run_steps(_job(_load(name))))
    assert f"--mode {mode}" in run
    for flag in ("--season", "--week"):
        for line in run.splitlines():
            if flag in line:
                assert f'"${flag.lstrip("-").upper()}"' in line, (
                    f"{name}: {flag} must come from the dispatch input, not a literal"
                )


@pytest.mark.parametrize("name", CAPTURE)
def test_the_dispatch_inputs_reach_the_script_through_env_not_interpolation(name):
    """``${{ }}`` spliced into ``run:`` is the shell-injection anti-pattern; env + an array is not."""
    job = _job(_load(name))
    step = next(s for s in job["steps"] if "collect.py" in s.get("run", ""))
    assert set(step["env"]) == {"WEEK", "SEASON"}
    assert "${{" not in step["run"]
    # Quoted array expansion: an unquoted "$ARGS" string would let an input inject a second flag.
    assert '"${ARGS[@]}"' in step["run"]


@pytest.mark.parametrize("name", CAPTURE)
def test_a_forward_only_failure_is_never_swallowed(name):
    """``collect.py`` exits 1 when an unrecoverable source fails -- the run must be allowed to go red."""
    workflow = _load(name)
    job = _job(workflow)
    assert "continue-on-error" not in workflow
    assert "continue-on-error" not in job
    assert not any("continue-on-error" in step for step in job["steps"])
    # Asserted against the executable shell, not the file text: every one of these strings also
    # appears in the comments explaining why it is absent, so matching raw text would pass on
    # nothing and fail on prose.
    assert "|| true" not in "\n".join(_run_steps(job))


@pytest.mark.parametrize("name", CAPTURE)
def test_a_green_skip_is_not_hunted_for_in_the_output(name):
    """An unresolvable postgame week prints "Skipping ..." and exits 0 by design (#16).

    Grepping the output for it -- the obvious way to "catch" a quiet run -- would turn the one
    correct no-op into a failure and train the operator to ignore red crons.
    """
    run = "\n".join(_run_steps(_job(_load(name))))
    assert "Skipping" not in run
    assert "grep" not in run


# --------------------------------------------------------------------------- where the rows land
@pytest.mark.parametrize("name", CAPTURE)
def test_the_lake_is_pointed_at_the_bucket_and_the_secrets_come_from_secrets(name):
    env = _job(_load(name))["env"]
    assert env["LAKE_BACKEND"] == "s3"
    for secret in REQUIRED_SECRETS:
        assert env[secret] == f"${{{{ secrets.{secret} }}}}", f"{name}: {secret} must not be inline"


@pytest.mark.parametrize("name", CAPTURE)
def test_a_run_that_is_not_pointed_at_the_bucket_refuses_to_start(name):
    """The one silent-loss path not closed in code.

    A mistyped ``LAKE_BACKEND`` raises in ``store.lake`` and a missing ``LAKE_S3_*`` raises
    ``S3ConfigError`` (#20), but an *absent* ``LAKE_BACKEND`` resolves to the local backend, which
    writes to the runner's disk and is discarded with it: a green cron capturing nothing, week after
    week, losing the forward-only rows each time. The guard has to run before the capture does.
    """
    steps = _job(_load(name))["steps"]
    guard = next(
        (i for i, s in enumerate(steps) if 'LAKE_BACKEND" != "s3"' in s.get("run", "")),
        None,
    )
    assert guard is not None, f"{name}: nothing stops a run against the local backend"
    capture = next(i for i, s in enumerate(steps) if "collect.py" in s.get("run", ""))
    assert guard < capture, "the guard must run before the capture, not after it"


@pytest.mark.parametrize("name", CAPTURE)
def test_nothing_is_committed_back_to_the_repo(name):
    """Rows go to the bucket. A commit step would be a rollback to the pre-#9 design.

    Checked three ways, because ``permissions: contents: read`` alone only makes a *push* fail --
    it does not stop someone adding a commit step and then widening the permission to match.
    """
    job = _job(_load(name))
    run = "\n".join(_run_steps(job))
    assert not any(f"git {verb}" in run for verb in ("add", "commit", "push"))
    # A third-party committer action would leave `run:` clean; actions/checkout is the only one here.
    assert {step["uses"].split("@")[0] for step in job["steps"] if "uses" in step} == {
        "actions/checkout", "actions/setup-python",
    }
    assert _load(name)["permissions"] == {"contents": "read"}


# --------------------------------------------------------------------------- cross-file invariant
def test_both_workflows_share_one_concurrency_group():
    """Mutual exclusion holds only because the two files spell the same literal string.

    ``store.write_snapshot`` reads a partition, merges, dedups and writes it back. Two runs on the
    same partition are a lost update. The schedules never collide, but a manual dispatch can land on
    a scheduled run -- and nothing else in the repo would notice one file's group being renamed.
    """
    groups = {name: _load(name)["concurrency"] for name in CAPTURE}
    assert len({g["group"] for g in groups.values()}) == 1, f"groups diverged: {groups}"
    for name, concurrency in groups.items():
        # Cancelling mid-capture throws away rows that cannot be re-fetched.
        assert concurrency["cancel-in-progress"] is False, name


def test_the_capture_workflows_do_not_share_the_dashboard_refresh_group():
    """``refresh.yml`` commits ``season.db`` and touches no lake partition -- it must not serialize
    behind a capture, and a capture must not wait on it."""
    assert _load("refresh.yml")["concurrency"]["group"] != _load("collect-prelock.yml")[
        "concurrency"
    ]["group"]


# --------------------------------------------------------------------------- the swap gate is human-only
def test_no_cron_regenerates_the_swap_gate():
    """The #34 default-source swap must stay a deliberate human act — no workflow regenerates it.

    ``projections.source.default_source`` reads ``src/model/fit/swap_gate.json``; regenerating it flips
    the default projection source for every surface in the tool. Its safety rests entirely on "committing
    the new state is a deliberate human act". ``refresh.yml`` is already a cron that commits data back to
    the repo, so that assumption is one workflow edit away from being false: adding ``eval_swap_gate.py``
    to any workflow would swap the tool's projections with no human in the loop. Pinned across **every**
    workflow (not only the two capture crons), the companion to
    ``test_nothing_is_committed_back_to_the_repo``.
    """
    for yml in WORKFLOWS.glob("*.yml"):
        text = yml.read_text(encoding="utf-8")
        assert "eval_swap_gate" not in text, f"{yml.name} regenerates the swap gate — it must be human-only"
        assert "swap_gate.json" not in text, f"{yml.name} writes the swap-gate artifact — it must be human-only"


# --------------------------------------------------------------------------- shared job shape
@pytest.mark.parametrize("name", CAPTURE)
def test_each_job_declares_a_timeout_and_installs_the_package_the_way_the_suite_does(name):
    job = _job(_load(name))
    assert isinstance(job.get("timeout-minutes"), int) and job["timeout-minutes"] >= 20
    install = "\n".join(_run_steps(job))
    assert "pip install -e ." in install
