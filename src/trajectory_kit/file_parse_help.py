from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterator, Any, Optional, Literal
import random 
import math


def iter_records(
    filepath: str | Path,
    *,
    mode: str,
    header_pred: Optional[Callable[[str], bool]] = None,
    count_from_header: Optional[Callable[[str], int]] = None,
    record_pred: Optional[Callable[[str], bool]] = None,
    parse_row: Callable[[str, int], Any],
    start_index: int = 0,
    encoding: str = "ascii",
    errors: str = "replace",
    ) -> Iterator[Any]:

    """
    A generic iterator for various types of .txt-like MD files. 
    Supports two modes of operation:
      - mode="counted": find a header, read N following lines
      - mode="predicate": yield every line matching record_pred

    parse_row(line, i) parses each yielded line; i is a running index starting at start_index.
    counted mode:
      - requires header_pred, count_from_header
    predicate mode:
      - requires record_pred

    Parameters:
    ----------
    filepath: str | Path
        The file path to the .txt-like file. This is required and should be a
        .txt, .psf, .pdb, etc.
    mode: str
        The mode of operation. Must be either "counted" or "predicate".
    header_pred: Optional[Callable[[str], bool]]
        A predicate function that takes a line as input and returns True if it is the header line
        that precedes the counted section. Required if mode is "counted".
    count_from_header: Optional[Callable[[str], int]]
        A function that takes the header line as input and returns the number of lines to read
        after the header. Required if mode is "counted".
    record_pred: Optional[Callable[[str], bool]]
        A predicate function that takes a line as input and returns True if the line should be yielded
        in predicate mode. Required if mode is "predicate".
    parse_row: Callable[[str, int], Any]
        A function that takes a line and its index as input and returns the parsed record. Required in both modes.
    start_index: int
        The starting index for the records. Default is 0.
    encoding: str
        The encoding to use when reading the file. Default is "ascii".
    errors: str
        The error handling strategy to use when reading the file. Default is "replace".
    
    Returns:
    -------
    Iterator[Any]
        An iterator that yields parsed records from the file according to the specified mode and predicates.
    """

    if mode not in ("counted", "predicate"):
        raise ValueError("mode must be 'counted' or 'predicate'.")

    with open(filepath, "rt", encoding=encoding, errors=errors) as f:

        # counted mode
        if mode == "counted":
            if header_pred is None or count_from_header is None:
                raise ValueError("counted mode requires header_pred and count_from_header.")
            for line in f:
                if header_pred(line):
                    n = count_from_header(line)
                    for i in range(start_index, start_index + n):
                        row = next(f)
                        yield parse_row(row, i)
                    return
            raise ValueError("Header not found for counted section.")

        # predicate mode
        if record_pred is None:
            raise ValueError("predicate mode requires record_pred.")

        i = start_index
        for line in f:
            if record_pred(line):
                yield parse_row(line, i)
                i += 1


def iter_records_sample(
    filepath,
    *,
    record_pred,
    parse_row,
    start_index=0,
    encoding="utf-8",
    errors="replace",
    target_sample_size=7000,
    rng_seed=None,
    ):

    '''
    Single-pass Bernoulli sampler with automatic probability selection.

    Behaviour
    ---------
    - Target sample size: `target_sample_size`
    - If file has <= target_sample_size lines → sample_probability = 1
    - Otherwise → sample_probability = target_sample_size / number_of_lines
    '''

    filepath = Path(filepath)
    rng = random.Random(rng_seed)

    # count lines  
    with open(filepath, "rt", encoding=encoding, errors=errors) as f:
        number_of_lines = sum(1 for _ in f)

    if number_of_lines == 0:
        return {
            "number_of_lines": 0,
            "number_of_sampled_lines": 0,
            "number_of_sampled_eligible_records": 0,
            "sampled_records": [],
        }

    # get probability
    if number_of_lines <= target_sample_size:
        sample_probability = 1.0
    else:
        sample_probability = target_sample_size / number_of_lines

    sampled_records = []
    number_of_sampled_lines = 0
    number_of_sampled_eligible_records = 0
    eligible_record_index = start_index

    # full sample
    if sample_probability >= 1.0:

        with open(filepath, "rt", encoding=encoding, errors=errors) as f:

            for line in f:

                number_of_sampled_lines += 1

                if record_pred(line):

                    sampled_records.append(
                        parse_row(line, eligible_record_index)
                    )

                    number_of_sampled_eligible_records += 1
                    eligible_record_index += 1

        return {
            "number_of_lines": number_of_lines,
            "number_of_sampled_lines": number_of_sampled_lines,
            "number_of_sampled_eligible_records": number_of_sampled_eligible_records,
            "sampled_records": sampled_records,
            "sampling_metadata": {
                "sampling_mode": "bernoulli_skip",
                "target_sample_size": target_sample_size,
                "sample_probability": sample_probability,
                "rng_seed": rng_seed,
            },
        }

    # geometric skip sample
    log_1mp = math.log1p(-sample_probability)

    u = max(rng.random(), 1e-16)
    next_sampled_line = int(math.log(u) / log_1mp)

    with open(filepath, "rt", encoding=encoding, errors=errors) as f:

        for physical_line_number, line in enumerate(f):

            is_sampled = (physical_line_number == next_sampled_line)

            if is_sampled:

                number_of_sampled_lines += 1

                u = max(rng.random(), 1e-16)
                gap = int(math.log(u) / log_1mp)

                next_sampled_line = physical_line_number + 1 + gap

            if record_pred(line):

                if is_sampled:

                    sampled_records.append(
                        parse_row(line, eligible_record_index)
                    )

                    number_of_sampled_eligible_records += 1

                eligible_record_index += 1

    return {
        "number_of_lines": number_of_lines,
        "number_of_sampled_lines": number_of_sampled_lines,
        "number_of_sampled_eligible_records": number_of_sampled_eligible_records,
        "sampled_records": sampled_records,
        "sampling_metadata": {
            "sampling_mode": "bernoulli_skip",
            "target_sample_size": target_sample_size,
            "sample_probability": sample_probability,
            "rng_seed": rng_seed,
        },
    }


def resolve_frame_interval(
    frame_inc,
    ) -> tuple[int, int | None, int]:

    """
    Convert a user-facing frame interval into Python range semantics:
        (start_inclusive, stop_exclusive, step)

    This is the shared implementation for all trajectory formats. The
    user-facing stop is always interpreted as **inclusive** — this is
    trajectory-kit's API contract. Internally stop is converted by +1
    so it can be used directly in range() and slice comparisons.

    Accepted input shapes
    ---------------------
    ()                  → (0, None, 1)   — all frames, step 1
    (start, stop)       → (start, stop+1, 1)
    (start, stop, step) → (start, stop+1, step)

    None is accepted for start or stop to mean "no bound":
        (None, 10)  → (0, 11, 1)
        (5, None)   → (5, None, 1)

    Parameters
    ----------
    frame_inc : tuple
        The raw frame_interval value from the query dictionary.

    Returns
    -------
    tuple[int, int | None, int]
        (start, stop, step) in Python range semantics.
        stop may be None meaning "read to end of file".

    Raises
    ------
    ValueError
        If frame_inc has the wrong length or step < 1.
    """

    if not frame_inc:
        return 0, None, 1

    if len(frame_inc) not in (2, 3):
        raise ValueError(
            "frame_interval must have 2 or 3 elements (start, stop[, step])."
        )

    a, b = frame_inc[:2]

    start = 0        if a is None else int(a)
    stop  = None     if b is None else (int(b) + 1)
    step  = int(frame_inc[2]) if len(frame_inc) == 3 else 1

    if step < 1:
        raise ValueError("frame_interval step must be >= 1.")

    return start, stop, step
