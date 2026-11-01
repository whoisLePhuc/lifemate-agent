<p align="center">
  <img src="Assets/logo.png" alt="Hephaestus" width="160">
</p>

<h1 align="center">LifeMate Agent</h1>

<p align="center">
  <strong>A Vietnamese voice-to-action agent for Zalo AI Challenge 2026:<br>
  understanding conversational context and completing tasks through the provided tools.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/focus-voice--to--action-007ec6" alt="Focus: voice-to-action">
  <img src="https://img.shields.io/badge/language-Vietnamese-4c1" alt="Language: Vietnamese">
  <img src="https://img.shields.io/badge/status-initialized-fe7d37" alt="Status: initialized">
</p>

## About

**LifeMate Agent** is team **Hephaestus**'s entry for the **LifeMate** challenge at **Zalo AI Challenge 2026**. The project aims to turn spoken Vietnamese requests into tool actions grounded in user context, conversation history, and tool results.

This repository hosts the team's competition solution, including implementation, configuration, evaluation, and submission materials as they become available.

## Competition

| Item | Details |
|---|---|
| Competition | Zalo AI Challenge 2026 |
| Challenge | LifeMate |
| Team | Hephaestus |
| Language | Vietnamese |
| Task | Voice-driven task completion through tool calls |

## Objectives

- **Speech recognition:** transcribe the current request and relevant user turns in conversation history.
- **Contextual reasoning:** resolve entities, references, and time expressions using the provided context.
- **Tool execution:** select available tools, validate arguments, and carry out tasks across multiple steps.
- **Clarification:** request missing information when needed and recognize requests beyond the available tools.
- **Reproducibility:** evaluate task completion consistently and prepare for offline inference under competition requirements.

## Project Status

The repository is in its initial setup stage. The objectives above describe the intended solution; a runnable agent, evaluation results, and submission package are not yet available.

Setup and usage instructions will be added alongside the implementation.

## Data and Submission

- Competition datasets and model weights are not included in this repository.
- The agent accesses application data through the provided tools, with gold labels excluded from its input.
- Predictions must follow the organizers' submission format, using executed tool calls and the final state hash returned by the environment.
- Data access, model usage, and submission packaging must follow the competition rules and applicable licenses.

This is a participant team's repository, not an official repository of the competition organizers.
