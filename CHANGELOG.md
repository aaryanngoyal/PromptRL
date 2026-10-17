# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-17

### Added
- Initial project structure with Q-learning prompt optimization
- Batched LLM inference via `eval_core` C++ shared library
- 13 prompt mutation actions in the action space
- Epsilon-greedy Q-learning agent with decay
- Batched reward calculator for 5-10x speedup
- C++ `eval_core` library wrapping `llama.cpp` for single-forward-pass evaluation
- `optimize_prompt.py` main entry point
