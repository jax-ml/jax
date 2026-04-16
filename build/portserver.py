#!/usr/bin/python3
#
# Copyright 2015 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""A server to hand out network ports to applications running on one host.

Typical usage:
 1) Run one instance of this process on each of your unittest farm hosts.
 2) Set the PORTSERVER_ADDRESS environment variable in your test runner
    environment to let the portpicker library know to use a port server
    rather than attempt to find ports on its own.

$ /path/to/portserver.py &
$ export PORTSERVER_ADDRESS=@unittest-portserver
$ # ... launch a bunch of unittest runners using portpicker ...
"""

import argparse
import asyncio
import collections
import logging
import signal
import socket
import sys
import psutil

log = None  # Initialized to a logging.Logger by _configure_logging().

_PROTOS = [(socket.SOCK_STREAM, socket.IPPROTO_TCP),
           (socket.SOCK_DGRAM, socket.IPPROTO_UDP)]


def _get_process_command_line(pid):
    try:
        return psutil.Process(pid).cmdline()
    except psutil.NoSuchProcess:
        return ''


def _get_process_start_time(pid):
    try:
        return psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return 0.0


# TODO: Consider importing portpicker.bind() instead of duplicating the code.
def _bind(port, socket_type, socket_proto):
    """Try to bind to a socket of the specified type, protocol, and port.

    For the port to be considered available, the kernel must support at least
    one of (IPv6, IPv4), and the port must be available on each supported
    family.

    Args:
      port: The port number to bind to, or 0 to have the OS pick a free port.
      socket_type: The type of the socket (ex: socket.SOCK_STREAM).
      socket_proto: The protocol of the socket (ex: socket.IPPROTO_TCP).

    Returns:
      The port number on success or None on failure.
    """
    got_socket = False
    for family in (socket.AF_INET6, socket.AF_INET):
        try:
            sock = socket.socket(family, socket_type, socket_proto)
            got_socket = True
        except OSError:
            continue
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', port))
            if socket_type == socket.SOCK_STREAM:
                sock.listen(1)
            port = sock.getsockname()[1]
        except OSError:
            return None
        finally:
            sock.close()
    return port if got_socket else None


def _is_port_free(port):
    """Check if specified port is free.

    Args:
      port: integer, port to check
    Returns:
      boolean, whether it is free to use for both TCP and UDP
    """
    return _bind(port, *_PROTOS[0]) and _bind(port, *_PROTOS[1])


def _should_allocate_port(pid):
    """Determine if we should allocate a port for use by the given process id."""
    if pid <= 0:
        log.info('Not allocating a port to invalid pid')
        return False
    if pid == 1:
        # The client probably meant to send us its parent pid but
        # had been reparented to init.
        log.info('Not allocating a port to init.')
        return False

    if not psutil.pid_exists(pid):
        log.info('Not allocating a port to a non-existent process')
        return False
    return True


async def _start_windows_server(client_connected_cb, path):
    """Start the server on Windows using named pipes."""
    def protocol_factory():
        stream_reader = asyncio.StreamReader()
        stream_reader_protocol = asyncio.StreamReaderProtocol(
            stream_reader, client_connected_cb)
        return stream_reader_protocol

    loop = asyncio.get_event_loop()
    server, *_ = await loop.start_serving_pipe(protocol_factory, address=path)

    return server


class _PortInfo:
    """Container class for information about a given port assignment.

    Attributes:
      port: integer port number
      pid: integer process id or 0 if unassigned.
      start_time: Time in seconds since the epoch that the process started.
    """

    __slots__ = ('port', 'pid', 'start_time')

    def __init__(self, port):
        self.port = port
        self.pid = 0
        self.start_time = 0.0


class _PortPool:
    """Manage available ports for processes.

    Ports are reclaimed when the reserving process exits and the reserved port
    is no longer in use.  Only ports which are free for both TCP and UDP will be
    handed out.  It is easier to not differentiate between protocols.

    The pool must be pre-seeded with add_port_to_free_pool() calls
    after which get_port_for_process() will allocate and reclaim ports.
    The len() of a _PortPool returns the total number of ports being managed.

    Attributes:
      ports_checked_for_last_request: The number of ports examined in order to
          return from the most recent get_port_for_process() request.  A high
          number here likely means the number of available ports with no active
          process using them is getting low.
    """

    def __init__(self):
        self._port_queue = collections.deque()
        self.ports_checked_for_last_request = 0

    def num_ports(self):
        return len(self._port_queue)

    def get_port_for_process(self, pid):
        """Allocates and returns port for pid or 0 if none could be allocated."""
        if not self._port_queue:
            raise RuntimeError('No ports being managed.')

        # Avoid an infinite loop if all ports are currently assigned.
        check_count = 0
        max_ports_to_test = len(self._port_queue)
        while check_count < max_ports_to_test:
            # Get the next candidate port and move it to the back of the queue.
            candidate = self._port_queue.pop()
            self._port_queue.appendleft(candidate)
            check_count += 1
            if (candidate.start_time == 0.0 or
                candidate.start_time != _get_process_start_time(candidate.pid)):
                if _is_port_free(candidate.port):
                    candidate.pid = pid
                    candidate.start_time = _get_process_start_time(pid)
                    if not candidate.start_time:
                        log.info("Can't read start time for pid %d.", pid)
                    self.ports_checked_for_last_request = check_count
                    return candidate.port
                else:
                    log.info(
                        'Port %d unexpectedly in use, last owning pid %d.',
                        candidate.port, candidate.pid)

        log.info('All ports in use.')
        self.ports_checked_for_last_request = check_count
        return 0

    def add_port_to_free_pool(self, port):
        """Add a new port to the free pool for allocation."""
        if port < 1 or port > 65535:
            raise ValueError(
                'Port must be in the [1, 65535] range, not %d.' % port)
        port_info = _PortInfo(port=port)
        self._port_queue.append(port_info)


class _PortServerRequestHandler:
    """A class to handle port allocation and status requests.

    Allocates ports to process ids via the dead simple port server protocol
    when the handle_port_request asyncio.coroutine handler has been registered.
    Statistics can be logged using the dump_stats method.
    """

    def __init__(self, ports_to_serve):
        """Initialize a new port server.

        Args:
          ports_to_serve: A sequence of unique port numbers to test and offer
              up to clients.
        """
        self._port_pool = _PortPool()
        self._total_allocations = 0
        self._denied_allocations = 0
        self._client_request_errors = 0
        for port in ports_to_serve:
            self._port_pool.add_port_to_free_pool(port)

    async def handle_port_request(self, reader, writer):
        client_data = await reader.read(100)
        self._handle_port_request(client_data, writer)
        writer.close()

    def _handle_port_request(self, client_data, writer):
        """Given a port request body, parse it and respond appropriately.

        Args:
          client_data: The request bytes from the client.
          writer: The asyncio Writer for the response to be written to.
        """
        try:
            if len(client_data) > 20:
                raise ValueError('More than 20 characters in "pid".')
            pid = int(client_data)
        except ValueError as error:
            self._client_request_errors += 1
            log.warning('Could not parse request: %s', error)
            return

        log.info('Request on behalf of pid %d.', pid)
        log.info('cmdline: %s', _get_process_command_line(pid))

        if not _should_allocate_port(pid):
            self._denied_allocations += 1
            return

        port = self._port_pool.get_port_for_process(pid)
        if port > 0:
            self._total_allocations += 1
            writer.write(f'{port:d}\n'.encode())
            log.debug('Allocated port %d to pid %d', port, pid)
        else:
            self._denied_allocations += 1

    def dump_stats(self):
        """Logs statistics of our operation."""
        log.info('Dumping statistics:')
        stats = []
        stats.append(
            f'client-request-errors {self._client_request_errors}')
        stats.append(f'denied-allocations {self._denied_allocations}')
        stats.append(f'num-ports-managed {self._port_pool.num_ports()}')
        stats.append('num-ports-checked-for-last-request {}'.format(
            self._port_pool.ports_checked_for_last_request))
        stats.append(f'total-allocations {self._total_allocations}')
        for stat in stats:
            log.info(stat)


def _parse_command_line():
    """Configure and parse our command line flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--portserver_static_pool',
        type=str,
        default='15000-24999',
        help='Comma separated N-P Range(s) of ports to manage (inclusive).')
    parser.add_argument(
        '--portserver_address',
        '--portserver_unix_socket_address', # Alias to be backward compatible
        type=str,
        default='@unittest-portserver',
        help='Address of AF_UNIX socket on which to listen on Unix (first @ is '
             'a NUL) or the name of the pipe on Windows (first @ is the '
             r'\\.\pipe\ prefix).')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose messages.')
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help='Enable full debug messages.')
    return parser.parse_args(sys.argv[1:])


def _parse_port_ranges(pool_str):
    """Given a 'N-P,X-Y' description of port ranges, return a set of ints."""
    ports = set()
    for range_str in pool_str.split(','):
        try:
            a, b = range_str.split('-', 1)
            start, end = int(a), int(b)
        except ValueError:
            log.error('Ignoring unparsable port range %r.', range_str)
            continue
        if start < 1 or end > 65535:
            log.error('Ignoring out of bounds port range %r.', range_str)
            continue
        ports.update(set(range(start, end + 1)))
    return ports


def _configure_logging(verbose=False, debug=False):
    """Configure the log global, message format, and verbosity settings."""
    overall_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format=('{levelname[0]}{asctime}.{msecs:03.0f} {thread} '
                '{filename}:{lineno}] {message}'),
        datefmt='%m%d %H:%M:%S',
        style='{',
        level=overall_level)
    global log
    log = logging.getLogger('portserver')
    # The verbosity controls our loggers logging level, not the global
    # one above. This avoids debug messages from libraries such as asyncio.
    log.setLevel(logging.DEBUG if verbose else overall_level)


async def async_main(config):
    ports_to_serve = _parse_port_ranges(config.portserver_static_pool)
    if not ports_to_serve:
        log.error('No ports.  Invalid port ranges in --portserver_static_pool?')
        sys.exit(1)

    request_handler = _PortServerRequestHandler(ports_to_serve)
    loop = asyncio.get_running_loop()

    if sys.platform == 'win32':
        # On Windows, we need to periodically pause the loop to allow the user
        # to send a break signal (e.g. ctrl+c)
        def listen_for_signal():
            loop.call_later(0.5, listen_for_signal)

        loop.call_later(0.5, listen_for_signal)

        path = config.portserver_address.replace('@', '\\\\.\\pipe\\', 1)
        server = await _start_windows_server(
            request_handler.handle_port_request,
            path=path)
    else:
        loop.add_signal_handler(
            signal.SIGUSR1, request_handler.dump_stats)

        server = await asyncio.start_unix_server(
            request_handler.handle_port_request,
            path=config.portserver_address.replace('@', '\0', 1))

    log.info('Serving on %s', config.portserver_address)

    try:
        await server.serve_forever()
    finally:
        server.close()
        await server.wait_closed()
        if sys.platform != 'win32':
             loop.remove_signal_handler(signal.SIGUSR1)
        request_handler.dump_stats()


def main():
    config = _parse_command_line()
    _configure_logging(verbose=config.verbose, debug=config.debug)
    try:
        asyncio.run(async_main(config))
    except KeyboardInterrupt:
        log.info('Stopping due to ^C.')
    log.info('Goodbye.')


if __name__ == '__main__':
    main()
