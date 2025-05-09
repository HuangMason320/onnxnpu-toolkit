# Changelog

All notable changes to this project are documented in this file.  
This project adheres to [Keep a Changelog](https://keepachangelog.com/) and [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Added
- Add onnx.checker for `onnxnpu check` and `onnxnpu opt`  command
- Add test code for onnx.checker's in [`tests/test_onnxchecker.py`](tests/test_onnxchecker.py) -> Add pytest package into project
### Changed

### Fixed
 

## [0.1.1] – 2025-05-08
### Added
- Add `Constant` into KL series' json file.
- `onnxnpu list` subcommand to list all built-in hardware profiles.

### Changed

### Fixed

---

## [0.1.0] – 2025-05-07
### Added
- Initial release, implemented:
  - Model structure summary & dynamic-axes detection
  - Compatibility scan against multiple hardware profiles
  - Markdown and native CLI report output
- `-V/--version` flag to display current version.