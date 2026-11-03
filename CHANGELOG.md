
# Changelog
## [2.0.0] - 2026-11-03

GridExpand is now one Python package (`gridexpand`) with one `uv` environment and one command, backed by the
`surrogrid` database schema. Start with [GridExpand/README.md](GridExpand/README.md). GridForecast is unchanged.

### Added
- `gridexpand run <run.yaml>`: Steps 2-4 and the expansion analyses of a region from one run YAML, with run
  directories, `--resume` and `gridexpand status` ([configuration](GridExpand/docs/configuration.md)).
- Scenario YAMLs with configurable electrification: adoption shares per building type, PV, battery, heat-pump and
  EV sizing rules, TEASER retrofit level, GHD rules, templates in `config/` ([method](GridExpand/docs/method.md)).
- PyPSA/linopy optimizer next to urbs (`--optimizer pypsa`) and a configurable solver with provenance
  ([Step 3](GridExpand/docs/steps/3_urbs.md)).
- Internal 1R1C space-heat model with HEMS preheating, heating-degree-day heat-pump sizing with a flexibility
  buffer, HEMS session caps for EV charging.
- Staged LV reinforcement rules with sourced costs: transformer exchange, load transfer, new stations, rONT
  ([expansion costs](GridExpand/docs/expansion_costs.md)).
- Paired and aligned validation of real DSO grids against their synthetic counterparts, and the status-quo test
  ([paired validation](GridExpand/docs/paired_validation.md)).
- HTTP API for the GridPlanner UI with a versioned contract, a Docker image and a GHCR workflow
  ([API](GridExpand/docs/api.md)).
- Versioned `surrogrid` schema migrations and maintenance commands, `gridexpand db ...`
  ([database](GridExpand/docs/database.md)).
- Unit tests without a database (`uv run pytest`) and an end-to-end regression harness on a sandbox database.

### Changed
- One package layout below `src/gridexpand/`; `gridexpand.paths` is the only module that knows directories.
  Runtime artifacts go to `work/`, static inputs to `data/` (`scripts/migrate_local_layout.py` moves old checkouts).
- Results are written to the `surrogrid` schema; the HDF5 hand-off files remain for GridForecast.
- Step 4 solves every timestep on one pandapower net, sets the LV busbar to 0.96 p.u. and rates transformers by
  the station rating of the pylovo grid.
- Building profiles are seeded per building, so model cases and paired runs share one stochastic realization.
- Documentation rewritten: README, configuration reference, method, one document per step
  ([index](GridExpand/docs/README.md)).

### Fixed
- Reactive-power sign convention in the power flow.
- Transformer rating of parallel units, heat generation for buildings without electricity demand, postcode parsing
  of the weather region, schema creation on a fresh database.

### Removed
- The per-step folders, environments and scripts of 1.0.0, and the dead code of the vendored urbs, emobpy and
  DistrictGenerator packages.

## [1.0.0] Initial Release - SurroGrid - 2025-12-15
### Added
- Initial setup for project.


<!-- ---

# Guidelines for Updating the Changelog
## [Version X.X.X] - YYYY-MM-DD
### Added
- Description of newly implemented features or functions, with a reference to the issue or MR number if applicable (e.g., `#42`).

### Changed
- Description of changes or improvements made to existing functionality, where relevant.

### Fixed
- Explanation of bugs or issues that have been resolved.
  
### Deprecated
- Note any features that are marked for future removal.

### Removed
- List of any deprecated features that have been fully removed.

---

## Example Entries

- **Added**: `Added feature to analyze time-series data from smart meters. Closes #10.`
- **Changed**: `Refined energy demand forecast model for better accuracy.`
- **Fixed**: `Resolved error in database connection handling in simulation module.`
- **Deprecated**: `Marked support for legacy data formats as deprecated.`
- **Removed**: `Removed deprecated API endpoints no longer in use.`

---

## Versioning Guidelines

This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):
- **Major** (X): Significant changes, likely with breaking compatibility.
- **Minor** (Y): New features that are backward-compatible.
- **Patch** (Z): Bug fixes and minor improvements.

**Example Versions**:
- **[2.1.0]** for a backward-compatible new feature.
- **[2.0.1]** for a minor fix that doesn’t break existing functionality.

## Best Practices

1. **One Entry per Change**: Each update, bug fix, or new feature should have its own entry.
2. **Be Concise**: Keep descriptions brief and informative.
3. **Link Issues or MRs**: Where possible, reference related issues or merge requests for easy tracking.
4. **Date Each Release**: Add the release date in `YYYY-MM-DD` format for each version.
5. **Organize Unreleased Changes**: Document ongoing changes under the `[Unreleased]` section, which can be merged into the next release version.
 -->
