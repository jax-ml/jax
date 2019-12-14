# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

from __future__ import absolute_import

import cgi
import logging
import mimetypes
import os
import re
import shutil
import sys

from pip._vendor import requests
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response
from pip._vendor.six import PY2
from pip._vendor.six.moves.urllib import parse as urllib_parse

from pip._internal.exceptions import HashMismatch, InstallationError
from pip._internal.models.index import PyPI
from pip._internal.network.session import PipSession
from pip._internal.utils.encoding import auto_decode
from pip._internal.utils.filesystem import copy2_fixed
from pip._internal.utils.misc import (
    ask_path_exists,
    backup_dir,
    consume,
    display_path,
    format_size,
    hide_url,
    path_to_display,
    rmtree,
    splitext,
)
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.typing import MYPY_CHECK_RUNNING
from pip._internal.utils.ui import DownloadProgressProvider
from pip._internal.utils.unpacking import unpack_file
from pip._internal.utils.urls import get_url_scheme
from pip._internal.vcs import vcs

if MYPY_CHECK_RUNNING:
    from typing import (
        IO, Callable, List, Optional, Text, Tuple,
    )

    from mypy_extensions import TypedDict

    from pip._internal.models.link import Link
    from pip._internal.utils.hashes import Hashes
    from pip._internal.vcs.versioncontrol import VersionControl

    if PY2:
        CopytreeKwargs = TypedDict(
            'CopytreeKwargs',
            {
                'ignore': Callable[[str, List[str]], List[str]],
                'symlinks': bool,
            },
            total=False,
        )
    else:
        CopytreeKwargs = TypedDict(
            'CopytreeKwargs',
            {
                'copy_function': Callable[[str, str], None],
                'ignore': Callable[[str, List[str]], List[str]],
                'ignore_dangling_symlinks': bool,
                'symlinks': bool,
            },
            total=False,
        )


__all__ = ['get_file_content',
           'unpack_vcs_link',
           'unpack_file_url',
           'unpack_http_url', 'unpack_url',
           'parse_content_disposition', 'sanitize_content_filename']


logger = logging.getLogger(__name__)


def get_file_content(url, comes_from=None, session=None):
    # type: (str, Optional[str], Optional[PipSession]) -> Tuple[str, Text]
    """Gets the content of a file; it may be a filename, file: URL, or
    http: URL.  Returns (location, content).  Content is unicode.

    :param url:         File path or url.
    :param comes_from:  Origin description of requirements.
    :param session:     Instance of pip.download.PipSession.
    """
    if session is None:
        raise TypeError(
            "get_file_content() missing 1 required keyword argument: 'session'"
        )

    scheme = get_url_scheme(url)

    if scheme in ['http', 'https']:
        # FIXME: catch some errors
        resp = session.get(url)
        resp.raise_for_status()
        return resp.url, resp.text

    elif scheme == 'file':
        if comes_from and comes_from.startswith('http'):
            raise InstallationError(
                'Requirements file %s references URL %s, which is local'
                % (comes_from, url))

        path = url.split(':', 1)[1]
        path = path.replace('\\', '/')
        match = _url_slash_drive_re.match(path)
        if match:
            path = match.group(1) + ':' + path.split('|', 1)[1]
        path = urllib_parse.unquote(path)
        if path.startswith('/'):
            path = '/' + path.lstrip('/')
        url = path

    try:
        with open(url, 'rb') as f:
            content = auto_decode(f.read())
    except IOError as exc:
        raise InstallationError(
            'Could not open requirements file: %s' % str(exc)
        )
    return url, content


_url_slash_drive_re = re.compile(r'/*([a-z])\|', re.I)


def unpack_vcs_link(link, location):
    # type: (Link, str) -> None
    vcs_backend = _get_used_vcs_backend(link)
    assert vcs_backend is not None
    vcs_backend.unpack(location, url=hide_url(link.url))


def _get_used_vcs_backend(link):
    # type: (Link) -> Optional[VersionControl]
    """
    Return a VersionControl object or None.
    """
    for vcs_backend in vcs.backends:
        if link.scheme in vcs_backend.schemes:
            return vcs_backend
    return None


def _progress_indicator(iterable, *args, **kwargs):
    return iterable


def _download_url(
    resp,  # type: Response
    link,  # type: Link
    content_file,  # type: IO
    hashes,  # type: Optional[Hashes]
    progress_bar  # type: str
):
    # type: (...) -> None
    try:
        total_length = int(resp.headers['content-length'])
    except (ValueError, KeyError, TypeError):
        total_length = 0

    cached_resp = getattr(resp, "from_cache", False)
    if logger.getEffectiveLevel() > logging.INFO:
        show_progress = False
    elif cached_resp:
        show_progress = False
    elif total_length > (40 * 1000):
        show_progress = True
    elif not total_length:
        show_progress = True
    else:
        show_progress = False

    show_url = link.show_url

    def resp_read(chunk_size):
        try:
            # Special case for urllib3.
            for chunk in resp.raw.stream(
                    chunk_size,
                    # We use decode_content=False here because we don't
                    # want urllib3 to mess with the raw bytes we get
                    # from the server. If we decompress inside of
                    # urllib3 then we cannot verify the checksum
                    # because the checksum will be of the compressed
                    # file. This breakage will only occur if the
                    # server adds a Content-Encoding header, which
                    # depends on how the server was configured:
                    # - Some servers will notice that the file isn't a
                    #   compressible file and will leave the file alone
                    #   and with an empty Content-Encoding
                    # - Some servers will notice that the file is
                    #   already compressed and will leave the file
                    #   alone and will add a Content-Encoding: gzip
                    #   header
                    # - Some servers won't notice anything at all and
                    #   will take a file that's already been compressed
                    #   and compress it again and set the
                    #   Content-Encoding: gzip header
                    #
                    # By setting this not to decode automatically we
                    # hope to eliminate problems with the second case.
                    decode_content=False):
                yield chunk
        except AttributeError:
            # Standard file-like object.
            while True:
                chunk = resp.raw.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def written_chunks(chunks):
        for chunk in chunks:
            content_file.write(chunk)
            yield chunk

    progress_indicator = _progress_indicator

    if link.netloc == PyPI.netloc:
        url = show_url
    else:
        url = link.url_without_fragment

    if show_progress:  # We don't show progress on cached responses
        progress_indicator = DownloadProgressProvider(progress_bar,
                                                      max=total_length)
        if total_length:
            logger.info("Downloading %s (%s)", url, format_size(total_length))
        else:
            logger.info("Downloading %s", url)
    elif cached_resp:
        logger.info("Using cached %s", url)
    else:
        logger.info("Downloading %s", url)

    downloaded_chunks = written_chunks(
        progress_indicator(
            resp_read(CONTENT_CHUNK_SIZE),
            CONTENT_CHUNK_SIZE
        )
    )
    if hashes:
        hashes.check_against_chunks(downloaded_chunks)
    else:
        consume(downloaded_chunks)


def _copy_file(filename, location, link):
    copy = True
    download_location = os.path.join(location, link.filename)
    if os.path.exists(download_location):
        response = ask_path_exists(
            'The file %s exists. (i)gnore, (w)ipe, (b)ackup, (a)abort' %
            display_path(download_location), ('i', 'w', 'b', 'a'))
        if response == 'i':
            copy = False
        elif response == 'w':
            logger.warning('Deleting %s', display_path(download_location))
            os.remove(download_location)
        elif response == 'b':
            dest_file = backup_dir(download_location)
            logger.warning(
                'Backing up %s to %s',
                display_path(download_location),
                display_path(dest_file),
            )
            shutil.move(download_location, dest_file)
        elif response == 'a':
            sys.exit(-1)
    if copy:
        shutil.copy(filename, download_location)
        logger.info('Saved %s', display_path(download_location))


def unpack_http_url(
    link,  # type: Link
    location,  # type: str
    download_dir=None,  # type: Optional[str]
    session=None,  # type: Optional[PipSession]
    hashes=None,  # type: Optional[Hashes]
    progress_bar="on"  # type: str
):
    # type: (...) -> None
    if session is None:
        raise TypeError(
            "unpack_http_url() missing 1 required keyword argument: 'session'"
        )

    with TempDirectory(kind="unpack") as temp_dir:
        # If a download dir is specified, is the file already downloaded there?
        already_downloaded_path = None
        if download_dir:
            already_downloaded_path = _check_download_dir(link,
                                                          download_dir,
                                                          hashes)

        if already_downloaded_path:
            from_path = already_downloaded_path
            content_type = mimetypes.guess_type(from_path)[0]
        else:
            # let's download to a tmp dir
            from_path, content_type = _download_http_url(link,
                                                         session,
                                                         temp_dir.path,
                                                         hashes,
                                                         progress_bar)

        # unpack the archive to the build dir location. even when only
        # downloading archives, they have to be unpacked to parse dependencies
        unpack_file(from_path, location, content_type)

        # a download dir is specified; let's copy the archive there
        if download_dir and not already_downloaded_path:
            _copy_file(from_path, download_dir, link)

        if not already_downloaded_path:
            os.unlink(from_path)


def _copy2_ignoring_special_files(src, dest):
    # type: (str, str) -> None
    """Copying special files is not supported, but as a convenience to users
    we skip errors copying them. This supports tools that may create e.g.
    socket files in the project source directory.
    """
    try:
        copy2_fixed(src, dest)
    except shutil.SpecialFileError as e:
        # SpecialFileError may be raised due to either the source or
        # destination. If the destination was the cause then we would actually
        # care, but since the destination directory is deleted prior to
        # copy we ignore all of them assuming it is caused by the source.
        logger.warning(
            "Ignoring special file error '%s' encountered copying %s to %s.",
            str(e),
            path_to_display(src),
            path_to_display(dest),
        )


def _copy_source_tree(source, target):
    # type: (str, str) -> None
    def ignore(d, names):
        # Pulling in those directories can potentially be very slow,
        # exclude the following directories if they appear in the top
        # level dir (and only it).
        # See discussion at https://github.com/pypa/pip/pull/6770
        return ['.tox', '.nox'] if d == source else []

    kwargs = dict(ignore=ignore, symlinks=True)  # type: CopytreeKwargs

    if not PY2:
        # Python 2 does not support copy_function, so we only ignore
        # errors on special file copy in Python 3.
        kwargs['copy_function'] = _copy2_ignoring_special_files

    shutil.copytree(source, target, **kwargs)


def unpack_file_url(
    link,  # type: Link
    location,  # type: str
    download_dir=None,  # type: Optional[str]
    hashes=None  # type: Optional[Hashes]
):
    # type: (...) -> None
    """Unpack link into location.

    If download_dir is provided and link points to a file, make a copy
    of the link file inside download_dir.
    """
    link_path = link.file_path
    # If it's a url to a local directory
    if link.is_existing_dir():
        if os.path.isdir(location):
            rmtree(location)
        _copy_source_tree(link_path, location)
        if download_dir:
            logger.info('Link is a directory, ignoring download_dir')
        return

    # If --require-hashes is off, `hashes` is either empty, the
    # link's embedded hash, or MissingHashes; it is required to
    # match. If --require-hashes is on, we are satisfied by any
    # hash in `hashes` matching: a URL-based or an option-based
    # one; no internet-sourced hash will be in `hashes`.
    if hashes:
        hashes.check_against_path(link_path)

    # If a download dir is specified, is the file already there and valid?
    already_downloaded_path = None
    if download_dir:
        already_downloaded_path = _check_download_dir(link,
                                                      download_dir,
                                                      hashes)

    if already_downloaded_path:
        from_path = already_downloaded_path
    else:
        from_path = link_path

    content_type = mimetypes.guess_type(from_path)[0]

    # unpack the archive to the build dir location. even when only downloading
    # archives, they have to be unpacked to parse dependencies
    unpack_file(from_path, location, content_type)

    # a download dir is specified and not already downloaded
    if download_dir and not already_downloaded_path:
        _copy_file(from_path, download_dir, link)


def unpack_url(
    link,  # type: Link
    location,  # type: str
    download_dir=None,  # type: Optional[str]
    session=None,  # type: Optional[PipSession]
    hashes=None,  # type: Optional[Hashes]
    progress_bar="on"  # type: str
):
    # type: (...) -> None
    """Unpack link.
       If link is a VCS link:
         if only_download, export into download_dir and ignore location
          else unpack into location
       for other types of link:
         - unpack into location
         - if download_dir, copy the file into download_dir
         - if only_download, mark location for deletion

    :param hashes: A Hashes object, one of whose embedded hashes must match,
        or HashMismatch will be raised. If the Hashes is empty, no matches are
        required, and unhashable types of requirements (like VCS ones, which
        would ordinarily raise HashUnsupported) are allowed.
    """
    # non-editable vcs urls
    if link.is_vcs:
        unpack_vcs_link(link, location)

    # file urls
    elif link.is_file:
        unpack_file_url(link, location, download_dir, hashes=hashes)

    # http urls
    else:
        if session is None:
            session = PipSession()

        unpack_http_url(
            link,
            location,
            download_dir,
            session,
            hashes=hashes,
            progress_bar=progress_bar
        )


def sanitize_content_filename(filename):
    # type: (str) -> str
    """
    Sanitize the "filename" value from a Content-Disposition header.
    """
    return os.path.basename(filename)


def parse_content_disposition(content_disposition, default_filename):
    # type: (str, str) -> str
    """
    Parse the "filename" value from a Content-Disposition header, and
    return the default filename if the result is empty.
    """
    _type, params = cgi.parse_header(content_disposition)
    filename = params.get('filename')
    if filename:
        # We need to sanitize the filename to prevent directory traversal
        # in case the filename contains ".." path parts.
        filename = sanitize_content_filename(filename)
    return filename or default_filename


def _download_http_url(
    link,  # type: Link
    session,  # type: PipSession
    temp_dir,  # type: str
    hashes,  # type: Optional[Hashes]
    progress_bar  # type: str
):
    # type: (...) -> Tuple[str, str]
    """Download link url into temp_dir using provided session"""
    target_url = link.url.split('#', 1)[0]
    try:
        resp = session.get(
            target_url,
            # We use Accept-Encoding: identity here because requests
            # defaults to accepting compressed responses. This breaks in
            # a variety of ways depending on how the server is configured.
            # - Some servers will notice that the file isn't a compressible
            #   file and will leave the file alone and with an empty
            #   Content-Encoding
            # - Some servers will notice that the file is already
            #   compressed and will leave the file alone and will add a
            #   Content-Encoding: gzip header
            # - Some servers won't notice anything at all and will take
            #   a file that's already been compressed and compress it again
            #   and set the Content-Encoding: gzip header
            # By setting this to request only the identity encoding We're
            # hoping to eliminate the third case. Hopefully there does not
            # exist a server which when given a file will notice it is
            # already compressed and that you're not asking for a
            # compressed file and will then decompress it before sending
            # because if that's the case I don't think it'll ever be
            # possible to make this work.
            headers={"Accept-Encoding": "identity"},
            stream=True,
        )
        resp.raise_for_status()
    except requests.HTTPError as exc:
        logger.critical(
            "HTTP error %s while getting %s", exc.response.status_code, link,
        )
        raise

    content_type = resp.headers.get('content-type', '')
    filename = link.filename  # fallback
    # Have a look at the Content-Disposition header for a better guess
    content_disposition = resp.headers.get('content-disposition')
    if content_disposition:
        filename = parse_content_disposition(content_disposition, filename)
    ext = splitext(filename)[1]  # type: Optional[str]
    if not ext:
        ext = mimetypes.guess_extension(content_type)
        if ext:
            filename += ext
    if not ext and link.url != resp.url:
        ext = os.path.splitext(resp.url)[1]
        if ext:
            filename += ext
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, 'wb') as content_file:
        _download_url(resp, link, content_file, hashes, progress_bar)
    return file_path, content_type


def _check_download_dir(link, download_dir, hashes):
    # type: (Link, str, Optional[Hashes]) -> Optional[str]
    """ Check download_dir for previously downloaded file with correct hash
        If a correct file is found return its path else None
    """
    download_path = os.path.join(download_dir, link.filename)

    if not os.path.exists(download_path):
        return None

    # If already downloaded, does its hash match?
    logger.info('File was already downloaded %s', download_path)
    if hashes:
        try:
            hashes.check_against_path(download_path)
        except HashMismatch:
            logger.warning(
                'Previously-downloaded file %s has bad hash. '
                'Re-downloading.',
                download_path
            )
            os.unlink(download_path)
            return None
    return download_path
