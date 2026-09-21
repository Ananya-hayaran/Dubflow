"""Exceptions used by the DubFlow pipeline."""
from __future__ import annotations


class PipelineError(RuntimeError):
    """A pipeline stage failed. The message is written for the end user."""


class ValidationError(PipelineError):
    """An artifact (file, JSON, audio, final MP4) failed validation."""


class TranslationError(PipelineError):
    """Translation could not be completed."""
