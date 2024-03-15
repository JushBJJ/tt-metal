# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import time
import pwd
import logging


class WaitLock:
    def __init__(self):
        # don't just use /tmp/filename because /tmp/ has the sticky bit set
        # to prevent removal of another user's files
        self.lock_file = "/tmp/tenstorrent/lock"
        # make sure the directory exists and is world-writable
        lock_dir = os.path.dirname(self.lock_file)
        try:
            os.makedirs(lock_dir, exist_ok=True)
            os.chmod(lock_dir, 0o777)
        except (OSError, PermissionError):
            pass

        self.pid = os.getpid()
        self.owns_lock = False
        self._acquire_lock()

    def _is_lock_valid(self, and_is_ours=False):
        """Check if the lock file exists and if the PID within it corresponds to a running process."""
        if not os.path.exists(self.lock_file):
            return False
        try:
            with open(self.lock_file, "r") as f:
                pid = int(f.read().strip())
            # Check if this PID process exists by sending signal 0 (no signal, just error checking)
            os.kill(pid, 0)
        except (ProcessLookupError, ValueError):
            # OSError if the process doesn't exist, ValueError if PID is not an integer
            return False
        except PermissionError:
            # Permission error if the process is true and belongs to someone else
            return True if and_is_ours == False else False

        if and_is_ours and pid != self.pid:
            return False
        else:
            return True

    def _acquire_lock(self):
        """Acquire the lock, waiting and retrying if necessary."""
        prev_user = None
        if self.owns_lock:
            logging.warning(f"WaitLock::._aquire_lock: lock already acquired, skipping")
            return

        while True:
            if not self._is_lock_valid():
                try:
                    # Attempt to acquire the lock
                    if os.path.exists(self.lock_file):
                        os.remove(self.lock_file)  # recreate with us as owner
                    with open(self.lock_file, "w") as f:
                        f.write(str(self.pid))
                    # Make the file world-writable in case our process dies
                    os.chmod(self.lock_file, 0o666)
                    # Mark it as owned by us
                    os.chown(self.lock_file, os.getuid(), os.getgid())
                    # Double-check if the lock was successfully acquired
                    if self._is_lock_valid(and_is_ours=True):
                        break
                except OSError as e:
                    print(f"Failed to acquire lock: {e}")
                    # If lock acquisition failed due to an OS error, assume another process is acquiring the lock
                    pass
            try:
                user = pwd.getpwuid(os.stat(self.lock_file).st_uid).pw_name
            except OSError:
                pass
            if user != prev_user:
                print(f"Waiting for lock to be released by {user}")
                prev_user = user
            time.sleep(1)
        self.owns_lock = True

    def release(self):
        """Release the lock."""
        if self.owns_lock:
            try:
                os.remove(self.lock_file)
            except OSError:
                pass
            self.owns_lock = False

    def __del__(self):
        """Delete the lock file upon destruction of the object if this process holds the lock."""
        self.release()


# Example usage
if __name__ == "__main__":
    lock = WaitLock()
    print(f"Lock acquired by PID: {os.getpid()}")
    # Do some work here
    # Lock will be automatically released when the program ends or the lock object is deleted
