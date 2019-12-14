"""Prepares a distribution for installation
"""

# The following comment should be removed at some point in the future.
# mypy: strict-optional=False
# mypy: disallow-untyped-defs=False

import logging
import os

from pip._vendor import requests

from pip._internal.distributions import (
    make_distribution_for_install_requirement,
)
from pip._internal.distributions.installed import InstalledDistribution
from pip._internal.download import unpack_url
from pip._internal.exceptions import (
    DirectoryUrlHashUnsupported,
    HashUnpinned,
    InstallationError,
    PreviousBuildDirError,
    VcsHashUnsupported,
)
from pip._internal.utils.compat import expanduser
from pip._internal.utils.hashes import MissingHashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.marker_files import write_delete_marker_file
from pip._internal.utils.misc import display_path, normalize_path
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Optional

    from pip._internal.distributions import AbstractDistribution
    from pip._internal.index import PackageFinder
    from pip._internal.network.session import PipSession
    from pip._internal.req.req_install import InstallRequirement
    from pip._internal.req.req_tracker import RequirementTracker

logger = logging.getLogger(__name__)


def _get_prepared_distribution(req, req_tracker, finder, build_isolation):
    """Prepare a distribution for installation.
    """
    abstract_dist = make_distribution_for_install_requirement(req)
    with req_tracker.track(req):
        abstract_dist.prepare_distribution_metadata(finder, build_isolation)
    return abstract_dist


class RequirementPreparer(object):
    """Prepares a Requirement
    """

    def __init__(
        self,
        build_dir,  # type: str
        download_dir,  # type: Optional[str]
        src_dir,  # type: str
        wheel_download_dir,  # type: Optional[str]
        progress_bar,  # type: str
        build_isolation,  # type: bool
        req_tracker  # type: RequirementTracker
    ):
        # type: (...) -> None
        super(RequirementPreparer, self).__init__()

        self.src_dir = src_dir
        self.build_dir = build_dir
        self.req_tracker = req_tracker

        # Where still-packed archives should be written to. If None, they are
        # not saved, and are deleted immediately after unpacking.
        if download_dir:
            download_dir = expanduser(download_dir)
        self.download_dir = download_dir

        # Where still-packed .whl files should be written to. If None, they are
        # written to the download_dir parameter. Separate to download_dir to
        # permit only keeping wheel archives for pip wheel.
        if wheel_download_dir:
            wheel_download_dir = normalize_path(wheel_download_dir)
        self.wheel_download_dir = wheel_download_dir

        # NOTE
        # download_dir and wheel_download_dir overlap semantically and may
        # be combined if we're willing to have non-wheel archives present in
        # the wheelhouse output by 'pip wheel'.

        self.progress_bar = progress_bar

        # Is build isolation allowed?
        self.build_isolation = build_isolation

    @property
    def _download_should_save(self):
        # type: () -> bool
        if not self.download_dir:
            return False

        if os.path.exists(self.download_dir):
            return True

        logger.critical('Could not find download directory')
        raise InstallationError(
            "Could not find or access download directory '%s'"
            % display_path(self.download_dir))

    def prepare_linked_requirement(
        self,
        req,  # type: InstallRequirement
        session,  # type: PipSession
        finder,  # type: PackageFinder
        require_hashes,  # type: bool
    ):
        # type: (...) -> AbstractDistribution
        """Prepare a requirement that would be obtained from req.link
        """
        assert req.link
        link = req.link

        # TODO: Breakup into smaller functions
        if link.scheme == 'file':
            path = link.file_path
            logger.info('Processing %s', display_path(path))
        else:
            logger.info('Collecting %s', req.req or req)

        with indent_log():
            # @@ if filesystem packages are not marked
            # editable in a req, a non deterministic error
            # occurs when the script attempts to unpack the
            # build directory
            req.ensure_has_source_dir(self.build_dir)
            # If a checkout exists, it's unwise to keep going.  version
            # inconsistencies are logged later, but do not fail the
            # installation.
            # FIXME: this won't upgrade when there's an existing
            # package unpacked in `req.source_dir`
            if os.path.exists(os.path.join(req.source_dir, 'setup.py')):
                raise PreviousBuildDirError(
                    "pip can't proceed with requirements '%s' due to a"
                    " pre-existing build directory (%s). This is "
                    "likely due to a previous installation that failed"
                    ". pip is being responsible and not assuming it "
                    "can delete this. Please delete it and try again."
                    % (req, req.source_dir)
                )

            # Now that we have the real link, we can tell what kind of
            # requirements we have and raise some more informative errors
            # than otherwise. (For example, we can raise VcsHashUnsupported
            # for a VCS URL rather than HashMissing.)
            if require_hashes:
                # We could check these first 2 conditions inside
                # unpack_url and save repetition of conditions, but then
                # we would report less-useful error messages for
                # unhashable requirements, complaining that there's no
                # hash provided.
                if link.is_vcs:
                    raise VcsHashUnsupported()
                elif link.is_existing_dir():
                    raise DirectoryUrlHashUnsupported()
                if not req.original_link and not req.is_pinned:
                    # Unpinned packages are asking for trouble when a new
                    # version is uploaded. This isn't a security check, but
                    # it saves users a surprising hash mismatch in the
                    # future.
                    #
                    # file:/// URLs aren't pinnable, so don't complain
                    # about them not being pinned.
                    raise HashUnpinned()

            hashes = req.hashes(trust_internet=not require_hashes)
            if require_hashes and not hashes:
                # Known-good hashes are missing for this requirement, so
                # shim it with a facade object that will provoke hash
                # computation and then raise a HashMissing exception
                # showing the user what the hash should be.
                hashes = MissingHashes()

            download_dir = self.download_dir
            if link.is_wheel and self.wheel_download_dir:
                # when doing 'pip wheel` we download wheels to a
                # dedicated dir.
                download_dir = self.wheel_download_dir

            try:
                unpack_url(
                    link, req.source_dir, download_dir,
                    session=session, hashes=hashes,
                    progress_bar=self.progress_bar
                )
            except requests.HTTPError as exc:
                logger.critical(
                    'Could not install requirement %s because of error %s',
                    req,
                    exc,
                )
                raise InstallationError(
                    'Could not install requirement %s because of HTTP '
                    'error %s for URL %s' %
                    (req, exc, link)
                )

            if link.is_wheel:
                if download_dir:
                    # When downloading, we only unpack wheels to get
                    # metadata.
                    autodelete_unpacked = True
                else:
                    # When installing a wheel, we use the unpacked
                    # wheel.
                    autodelete_unpacked = False
            else:
                # We always delete unpacked sdists after pip runs.
                autodelete_unpacked = True
            if autodelete_unpacked:
                write_delete_marker_file(req.source_dir)

            abstract_dist = _get_prepared_distribution(
                req, self.req_tracker, finder, self.build_isolation,
            )

            if self._download_should_save:
                # Make a .zip of the source_dir we already created.
                if link.is_vcs:
                    req.archive(self.download_dir)
        return abstract_dist

    def prepare_editable_requirement(
        self,
        req,  # type: InstallRequirement
        require_hashes,  # type: bool
        use_user_site,  # type: bool
        finder  # type: PackageFinder
    ):
        # type: (...) -> AbstractDistribution
        """Prepare an editable requirement
        """
        assert req.editable, "cannot prepare a non-editable req as editable"

        logger.info('Obtaining %s', req)

        with indent_log():
            if require_hashes:
                raise InstallationError(
                    'The editable requirement %s cannot be installed when '
                    'requiring hashes, because there is no single file to '
                    'hash.' % req
                )
            req.ensure_has_source_dir(self.src_dir)
            req.update_editable(not self._download_should_save)

            abstract_dist = _get_prepared_distribution(
                req, self.req_tracker, finder, self.build_isolation,
            )

            if self._download_should_save:
                req.archive(self.download_dir)
            req.check_if_exists(use_user_site)

        return abstract_dist

    def prepare_installed_requirement(
        self,
        req,  # type: InstallRequirement
        require_hashes,  # type: bool
        skip_reason  # type: str
    ):
        # type: (...) -> AbstractDistribution
        """Prepare an already-installed requirement
        """
        assert req.satisfied_by, "req should have been satisfied but isn't"
        assert skip_reason is not None, (
            "did not get skip reason skipped but req.satisfied_by "
            "is set to %r" % (req.satisfied_by,)
        )
        logger.info(
            'Requirement %s: %s (%s)',
            skip_reason, req, req.satisfied_by.version
        )
        with indent_log():
            if require_hashes:
                logger.debug(
                    'Since it is already installed, we are trusting this '
                    'package without checking its hash. To ensure a '
                    'completely repeatable environment, install into an '
                    'empty virtualenv.'
                )
            abstract_dist = InstalledDistribution(req)

        return abstract_dist
