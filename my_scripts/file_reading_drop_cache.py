#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
import json
import itertools
from typing import Iterator
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info_rank0 = logger.info

def _iter_dynamic_jsonl(file_path: str, is_drop_cache: bool = False) -> Iterator[dict]:
    """
    Yield each JSONL line as a parsed dict, logging open and per-line latency.

    Each time this generator is re-iterated, the file is reopened from the start.
    If `is_drop_cache` is True, the file's entire page cache is dropped on open
    and fadvise(DONTNEED) is called on the just-read range after each line.
    """

    logger.info_rank0(f"[STREAM] START file_path={file_path}, is_drop_cache={is_drop_cache}")

    t_start = time.perf_counter()

    with open(file_path, "rb") as f:

        line_no = 0
        last_pos = 0
        cost_io_total = 0.0
        cost_fadv_total = 0.0

        fd = f.fileno()

        if is_drop_cache:
            t_fadv_start = time.perf_counter()

            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)

            t_fadv_end = time.perf_counter()
            cost_fadv = t_fadv_end - t_fadv_start
            cost_fadv_total += cost_fadv

            logger.info_rank0(f"[STREAM] drop_cache full file cost={cost_fadv * 1000:.2f}ms")

        while True:

            t_io_start = time.perf_counter()

            raw_line = f.readline()

            t_io_end = time.perf_counter()
            cost_io = t_io_end - t_io_start
            cost_io_total += cost_io

            if not raw_line:
                break # while True

            if line_no % 20 == 0:
                t_now = time.perf_counter()
                passed_from_start = t_now - t_start
                logger.info_rank0(
                    f"[STREAM] line_no={line_no}"
                    f" passed_from_start={passed_from_start * 1000:.2f}ms "
                    f" cost_io_total={cost_io_total * 1000:.2f}ms "
                    f" cost_fadv_total={cost_fadv_total * 1000:.2f}ms"
                )

            line = raw_line.decode("utf-8").strip()

            line_no += 1
            if not line:
                continue

            yield json.loads(line)

            if is_drop_cache:
                
                cur_pos = f.tell()
                last_pos = cur_pos

                t_fadv_start = time.perf_counter()
                #os.posix_fadvise(fd, last_pos, cur_pos - last_pos, os.POSIX_FADV_DONTNEED)
                # temp test
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                t_fadv_end = time.perf_counter()
                cost_fadv = t_fadv_end - t_fadv_start
                cost_fadv_total += cost_fadv

        t_now = time.perf_counter()
        passed_from_start = t_now - t_start
        logger.info_rank0(
            f"[STREAM] END"
            f" passed_from_start={passed_from_start * 1000:.2f}ms "
            f" cost_io_total={cost_io_total * 1000:.2f}ms "
            f" cost_fadv_total={cost_fadv_total * 1000:.2f}ms"
        )

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python file_reading_drop_cache.py <file_path> <is_drop_cache>")

    file_path = sys.argv[1]
    is_drop_cache = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False

    for _ in itertools.islice(_iter_dynamic_jsonl(file_path, is_drop_cache), 800):
        pass

if __name__ == "__main__":
    main()