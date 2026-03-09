"""Compatibility layer for optional d4rl offline env support."""

try:
    from d4rl import offline_env as offline_env  # type: ignore
except Exception:
    class _OfflineEnv:
        def __init__(self, *args, **kwargs):
            self.dataset_url = kwargs.get("dataset_url", None)

        def get_dataset(self, *args, **kwargs):
            raise ImportError(
                "d4rl is optional and not installed. Install d4rl to use offline dataset APIs."
            )

    class _OfflineEnvModule:
        OfflineEnv = _OfflineEnv

    offline_env = _OfflineEnvModule()
