# Changelog

All notable changes to TRINITY will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — Unreleased

First public release.

### Features

- Feedback-driven HII-region evolution code with phase transitions
  (energy-driven → transition → momentum-driven) and stopping fates
  (stall, dissolution, escape).
- Cloud density profiles: power-law, Bonnor-Ebert, homogeneous.
- Parameter sweep mode (Cartesian product and explicit-tuple modes).
- CLOUDY input-deck generation from TRINITY snapshots.
- Bundled minimal defaults (SB99 SPS table + cooling tables under
  `lib/default/`) so the README quickstart runs out of the box.
