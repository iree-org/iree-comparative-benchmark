## Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import socket
import subprocess

from contextlib import contextmanager, closing
from typing import Generator

from google.cloud import bigquery
from google.api_core.client_options import ClientOptions


def _reserve_local_port() -> int:
  """ Reserves a local TCP port and tricks it into the TIME_WAIT state, then closes it.
  That means the port will be reserved for the length to 2x TCP_FIN timeout (usually 60s on Linux)
  and can only be reused by this process or subprocesses during that time.

  This is a standard trick when you want a program listening on an ephemeral port,
  but this program requires you to specify a port on startup.
  """
  with closing(socket.socket()) as listen_socket:
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(("127.0.0.1", 0))
    listen_socket.listen(
        1
    )  # The `1` is the backlog - the size of the queue of unaccepted connections

    socketname = listen_socket.getsockname()

    with closing(socket.socket()) as connecting_socket:
      connecting_socket.connect(socketname)
      connected_socket, _ = listen_socket.accept()
      with closing(connected_socket):
        # On this return all 3 sockets will be closed by the context managers.
        # That means at the time we exit this function the ephemeral port will be
        # in the TIME_WAIT state and can be reopened by this or a sub-process using
        # the SO_REUSEADDR flag.
        return socketname[1]  # That's the ephemeral port


def is_bigquery_emulator_available() -> bool:
  """ Just checks if the bigquery-emulator is installed. """
  try:
    with subprocess.Popen(["bigquery-emulator", "--version"],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL) as process:
      return process.wait(10) == 0
  except:
    return False


@contextmanager
def emulate_bigquery(
    project_name: str,
    dataset_name: str) -> Generator[bigquery.Client, None, None]:
  """Returns a BigQuery Client that is connected to a temporary database which is provided by the bigquery-emulator."""
  http_port = _reserve_local_port()
  grpc_port = _reserve_local_port()

  with subprocess.Popen([
      "bigquery-emulator", f"--project={project_name}",
      f"--dataset={dataset_name}", f"--port={http_port}",
      f"--grpc-port={grpc_port}"
  ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL) as process:
    try:
      client_options = ClientOptions(api_endpoint=f"http://0.0.0.0:{http_port}")
      yield bigquery.Client(project=project_name, client_options=client_options)
    finally:
      process.terminate()
      try:
        process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        pass
      process.kill()
