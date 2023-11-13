#!/usr/bin/env python3
#
# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import requests
import socket
import struct
import time

ADB_SERVER_ADDR = ("localhost", 5037)


def adb_download_and_push_file(source_url: str,
                               destination: str,
                               verbose: bool = False):
  """Fetch file from the URL and stream to the device.
  In the case of fetching, this method avoids the temporary file on the host
  and reduces the overhead when the file is large.
  Args:
    source_url: URL to fetch the file.
    destination: the full destination path on the device.
    verbose: output verbose message.
  Returns:
    File path on the device.
  """

  if verbose:
    print(f"Streaming file {source_url} to {destination}.")

  req = requests.get(source_url, stream=True, timeout=60)
  if not req.ok:
    raise RuntimeError(
        f"Failed to fetch {source_url}: {req.status_code} - {req.text}")

  # Implement the ADB sync protocol to stream file chunk to the device, since
  # the adb client tool doesn't support it.
  #
  # Alternatively we can use thrid-party library such as
  # https://github.com/JeffLIrion/adb_shell. But the protocol we need is
  # simple and fairly stable. This part can be replaced with other solutions
  # if needed.
  #
  # To understand the details of the protocol, see
  # https://cs.android.com/android/_/android/platform/packages/modules/adb/+/93c8e3c26e4de3a2b767a2394200bc0721bb1e24:OVERVIEW.TXT

  def wait_ack_ok(sock: socket.socket):
    buf = bytearray()
    while len(buf) < 4:
      data = sock.recv(4 - len(buf))
      if not data:
        break
      buf += data

    if buf.decode("utf-8") != "OKAY":
      raise RuntimeError(f"ADB communication error: {buf}")

  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect(ADB_SERVER_ADDR)
    # Connect to any device (the first 4 hexadecimals is the following text
    # command length).
    sock.sendall(b"0012host:transport-any")
    wait_ack_ok(sock)
    # Switch to sync mode.
    sock.sendall(b"0005sync:")
    wait_ack_ok(sock)
    # Send the destination file path and file permissions 0755 (rwxr-xr-x).
    file_attr = f"{destination},{0o755}".encode("utf-8")
    sock.sendall(b"SEND" + struct.pack("I", len(file_attr)) + file_attr)
    # Stream the file chunks. 64k bytes is the max chunk size for adb.
    for data in req.iter_content(chunk_size=64 * 1024):
      sock.sendall(b"DATA" + struct.pack("I", len(data)) + data)
    # End the file stream and set the creation time.
    sock.sendall(b"DONE" + struct.pack("I", int(time.time())))
    wait_ack_ok(sock)

  return destination


def _parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Runs benchmarks.")
  parser.add_argument("-s",
                      "--source_url",
                      type=str,
                      required=True,
                      help="The url of file to download.")
  parser.add_argument("-d",
                      "--destination",
                      type=str,
                      required=True,
                      help="The path on the device to stream the file to.")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Show verbose messages.")
  return parser.parse_args()


if __name__ == "__main__":
  adb_download_and_push_file(**vars(_parse_arguments()))
