from astropy.cosmology import default_cosmology


class _CosmologyProxy:
    """Proxy to always get the current default cosmology.

    This allows the module to respect runtime changes to the default cosmology
    via `astropy.cosmology.default_cosmology.set()`.

    """

    def __getattr__(self, name):
        return getattr(default_cosmology.get(), name)


cosmo = _CosmologyProxy()
