"""Shared benchmark-generation code and utilities used by both Phase 1 and Phase 2.

Anything here must stay generation-agnostic: it defines the *fair, shared*
dataset and evaluation infrastructure both phases are measured against. If a
piece of code is specific to one phase's methodology, it belongs in that
phase's own package instead, not here.
"""
