"""Level 1 broadcast possession pipeline — team A/B/contested timeline."""

__all__ = ["run_pipeline"]


def run_pipeline(*args, **kwargs):
    from polyfut_video.main import run_pipeline as _run
    return _run(*args, **kwargs)
