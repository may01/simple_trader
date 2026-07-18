"""azlib — reusable, pure, pytest-tested logic for the action_zones experiment.

Notebooks under notebooks/action_zones/nb/ do orchestration only (load data,
call into azlib, save artifacts); every piece of logic that needs a unit
test lives in this package instead. See
external/docs/superpowers/plans/2026-07-19-zone-selection-experiment.md for
the full task breakdown (Task 0 = this scaffold; azlib gains its first real
module in Task 1).
"""
