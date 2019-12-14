# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

from pip._internal.distributions.base import AbstractDistribution


class InstalledDistribution(AbstractDistribution):
    """Represents an installed package.

    This does not need any preparation as the required information has already
    been computed.
    """

    def get_pkg_resources_distribution(self):
        return self.req.satisfied_by

    def prepare_distribution_metadata(self, finder, build_isolation):
        pass
