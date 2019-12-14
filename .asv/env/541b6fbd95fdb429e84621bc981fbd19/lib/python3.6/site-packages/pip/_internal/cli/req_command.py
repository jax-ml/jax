"""Contains the Command base classes that depend on PipSession.

The classes in this module are in a separate module so the commands not
needing download / PackageFinder capability don't unnecessarily import the
PackageFinder machinery and all its vendored dependencies, etc.
"""

# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

import os
from functools import partial

from pip._internal.cli.base_command import Command
from pip._internal.cli.command_context import CommandContextMixIn
from pip._internal.exceptions import CommandError
from pip._internal.index import PackageFinder
from pip._internal.legacy_resolve import Resolver
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import (
    install_req_from_editable,
    install_req_from_line,
    install_req_from_req_string,
)
from pip._internal.req.req_file import parse_requirements
from pip._internal.self_outdated_check import (
    make_link_collector,
    pip_self_version_check,
)
from pip._internal.utils.misc import normalize_path
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from optparse import Values
    from typing import List, Optional, Tuple
    from pip._internal.cache import WheelCache
    from pip._internal.models.target_python import TargetPython
    from pip._internal.req.req_set import RequirementSet
    from pip._internal.req.req_tracker import RequirementTracker
    from pip._internal.utils.temp_dir import TempDirectory


class SessionCommandMixin(CommandContextMixIn):

    """
    A class mixin for command classes needing _build_session().
    """
    def __init__(self):
        super(SessionCommandMixin, self).__init__()
        self._session = None  # Optional[PipSession]

    @classmethod
    def _get_index_urls(cls, options):
        """Return a list of index urls from user-provided options."""
        index_urls = []
        if not getattr(options, "no_index", False):
            url = getattr(options, "index_url", None)
            if url:
                index_urls.append(url)
        urls = getattr(options, "extra_index_urls", None)
        if urls:
            index_urls.extend(urls)
        # Return None rather than an empty list
        return index_urls or None

    def get_default_session(self, options):
        # type: (Values) -> PipSession
        """Get a default-managed session."""
        if self._session is None:
            self._session = self.enter_context(self._build_session(options))
        return self._session

    def _build_session(self, options, retries=None, timeout=None):
        # type: (Values, Optional[int], Optional[int]) -> PipSession
        session = PipSession(
            cache=(
                normalize_path(os.path.join(options.cache_dir, "http"))
                if options.cache_dir else None
            ),
            retries=retries if retries is not None else options.retries,
            trusted_hosts=options.trusted_hosts,
            index_urls=self._get_index_urls(options),
        )

        # Handle custom ca-bundles from the user
        if options.cert:
            session.verify = options.cert

        # Handle SSL client certificate
        if options.client_cert:
            session.cert = options.client_cert

        # Handle timeouts
        if options.timeout or timeout:
            session.timeout = (
                timeout if timeout is not None else options.timeout
            )

        # Handle configured proxies
        if options.proxy:
            session.proxies = {
                "http": options.proxy,
                "https": options.proxy,
            }

        # Determine if we can prompt the user for authentication or not
        session.auth.prompting = not options.no_input

        return session


class IndexGroupCommand(Command, SessionCommandMixin):

    """
    Abstract base class for commands with the index_group options.

    This also corresponds to the commands that permit the pip version check.
    """

    def handle_pip_version_check(self, options):
        # type: (Values) -> None
        """
        Do the pip version check if not disabled.

        This overrides the default behavior of not doing the check.
        """
        # Make sure the index_group options are present.
        assert hasattr(options, 'no_index')

        if options.disable_pip_version_check or options.no_index:
            return

        # Otherwise, check if we're using the latest version of pip available.
        session = self._build_session(
            options,
            retries=0,
            timeout=min(5, options.timeout)
        )
        with session:
            pip_self_version_check(session, options)


class RequirementCommand(IndexGroupCommand):

    @staticmethod
    def make_requirement_preparer(
        temp_build_dir,           # type: TempDirectory
        options,                  # type: Values
        req_tracker,              # type: RequirementTracker
        download_dir=None,        # type: str
        wheel_download_dir=None,  # type: str
    ):
        # type: (...) -> RequirementPreparer
        """
        Create a RequirementPreparer instance for the given parameters.
        """
        temp_build_dir_path = temp_build_dir.path
        assert temp_build_dir_path is not None
        return RequirementPreparer(
            build_dir=temp_build_dir_path,
            src_dir=options.src_dir,
            download_dir=download_dir,
            wheel_download_dir=wheel_download_dir,
            progress_bar=options.progress_bar,
            build_isolation=options.build_isolation,
            req_tracker=req_tracker,
        )

    @staticmethod
    def make_resolver(
        preparer,                            # type: RequirementPreparer
        session,                             # type: PipSession
        finder,                              # type: PackageFinder
        options,                             # type: Values
        wheel_cache=None,                    # type: Optional[WheelCache]
        use_user_site=False,                 # type: bool
        ignore_installed=True,               # type: bool
        ignore_requires_python=False,        # type: bool
        force_reinstall=False,               # type: bool
        upgrade_strategy="to-satisfy-only",  # type: str
        use_pep517=None,                     # type: Optional[bool]
        py_version_info=None            # type: Optional[Tuple[int, ...]]
    ):
        # type: (...) -> Resolver
        """
        Create a Resolver instance for the given parameters.
        """
        make_install_req = partial(
            install_req_from_req_string,
            isolated=options.isolated_mode,
            wheel_cache=wheel_cache,
            use_pep517=use_pep517,
        )
        return Resolver(
            preparer=preparer,
            session=session,
            finder=finder,
            make_install_req=make_install_req,
            use_user_site=use_user_site,
            ignore_dependencies=options.ignore_dependencies,
            ignore_installed=ignore_installed,
            ignore_requires_python=ignore_requires_python,
            force_reinstall=force_reinstall,
            upgrade_strategy=upgrade_strategy,
            py_version_info=py_version_info
        )

    def populate_requirement_set(
        self,
        requirement_set,  # type: RequirementSet
        args,             # type: List[str]
        options,          # type: Values
        finder,           # type: PackageFinder
        session,          # type: PipSession
        wheel_cache,      # type: Optional[WheelCache]
    ):
        # type: (...) -> None
        """
        Marshal cmd line args into a requirement set.
        """
        # NOTE: As a side-effect, options.require_hashes and
        #       requirement_set.require_hashes may be updated

        for filename in options.constraints:
            for req_to_add in parse_requirements(
                    filename,
                    constraint=True, finder=finder, options=options,
                    session=session, wheel_cache=wheel_cache):
                req_to_add.is_direct = True
                requirement_set.add_requirement(req_to_add)

        for req in args:
            req_to_add = install_req_from_line(
                req, None, isolated=options.isolated_mode,
                use_pep517=options.use_pep517,
                wheel_cache=wheel_cache
            )
            req_to_add.is_direct = True
            requirement_set.add_requirement(req_to_add)

        for req in options.editables:
            req_to_add = install_req_from_editable(
                req,
                isolated=options.isolated_mode,
                use_pep517=options.use_pep517,
                wheel_cache=wheel_cache
            )
            req_to_add.is_direct = True
            requirement_set.add_requirement(req_to_add)

        for filename in options.requirements:
            for req_to_add in parse_requirements(
                    filename,
                    finder=finder, options=options, session=session,
                    wheel_cache=wheel_cache,
                    use_pep517=options.use_pep517):
                req_to_add.is_direct = True
                requirement_set.add_requirement(req_to_add)
        # If --require-hashes was a line in a requirements file, tell
        # RequirementSet about it:
        requirement_set.require_hashes = options.require_hashes

        if not (args or options.editables or options.requirements):
            opts = {'name': self.name}
            if options.find_links:
                raise CommandError(
                    'You must give at least one requirement to %(name)s '
                    '(maybe you meant "pip %(name)s %(links)s"?)' %
                    dict(opts, links=' '.join(options.find_links)))
            else:
                raise CommandError(
                    'You must give at least one requirement to %(name)s '
                    '(see "pip help %(name)s")' % opts)

    def _build_package_finder(
        self,
        options,               # type: Values
        session,               # type: PipSession
        target_python=None,    # type: Optional[TargetPython]
        ignore_requires_python=None,  # type: Optional[bool]
    ):
        # type: (...) -> PackageFinder
        """
        Create a package finder appropriate to this requirement command.

        :param ignore_requires_python: Whether to ignore incompatible
            "Requires-Python" values in links. Defaults to False.
        """
        link_collector = make_link_collector(session, options=options)
        selection_prefs = SelectionPreferences(
            allow_yanked=True,
            format_control=options.format_control,
            allow_all_prereleases=options.pre,
            prefer_binary=options.prefer_binary,
            ignore_requires_python=ignore_requires_python,
        )

        return PackageFinder.create(
            link_collector=link_collector,
            selection_prefs=selection_prefs,
            target_python=target_python,
        )
