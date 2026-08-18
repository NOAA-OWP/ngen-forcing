"""General utilities"""

import json
import logging
import typing
import uuid
from collections import OrderedDict

import numpy as np

JSON_NOT_SERIALIZABLE_SENTINEL = "ERR_NOT_JSON_SERIALIZABLE"
JSON_NOT_SERIALIZABLE_FORMAT = JSON_NOT_SERIALIZABLE_SENTINEL + ":TYPE:{typ}"


class ExpectVsActualError(Exception):
    """Raised by assert_equal_with_tol"""


def serializer_with_fallback(obj: typing.Any):
    """Serializer for json.dump to handle typical types, numpy types, and non-serializable types,
    which are converted to a string composed of a sentinel and the type as the suffix.
    To be used as the `default=` parameter when calling json dump/dumps.
    Not to be called directly.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    # It is not serializable
    return JSON_NOT_SERIALIZABLE_FORMAT.format(typ=str(type(obj)))


def serialize_to_json(
    obj: typing.Any,
    out_file: str = None,
    sort_keys: bool = False,
    keep_keys: tuple = None,
) -> str:
    """Serialize the provided object.
    Parameters:
        out_file: optionally write it to a new file.
        sort_keys: optionally sort the keys (passed to json.dumps kwarg sort_keys).
        keep_keys: optionally filter it to keep only the keep_keys.
    Returns:
        A JSON string representation of the object.
    """
    dump_kwargs = {
        "default": serializer_with_fallback,
        "indent": 2,
        "sort_keys": sort_keys,
    }
    json_str = json.dumps(obj, **dump_kwargs)

    # Optionally filter
    if keep_keys:
        tmp = json.loads(json_str)
        tmp = {k: v for k, v in tmp.items() if k in keep_keys}
        json_str = json.dumps(tmp, **dump_kwargs)
        del tmp

    # Optionally write to file
    if out_file is not None:
        logging.info(f"Writing: {out_file}")
        with open(out_file, "w") as f:
            f.write(json_str)

    return json_str


def assert_equal_with_tol(
    expect: dict,
    actual: dict,
    keys_to_check: tuple | None = None,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-10,
    new_keys_in_actual_ok: bool = False,
):
    """Assert that the key,value pairs in `expect` have matching key,value pairs in `actual`, with numerical tolerance.
    It is okay if actual has extra keys that are not present in expect.
    If keys_to_check is defined, then only those keys will be checked.
    Raises ExpectVsActualError.
    """
    errors: list[Exception] = []

    if not keys_to_check:
        keys_to_check = list(expect.keys())

    logging.info(
        f"Asserting equality with absolute tolerance {absolute_tolerance} and relative tolerance {relative_tolerance} for {len(keys_to_check)} keys: {keys_to_check}"
    )

    keys_missing_from_actual = set(keys_to_check) - set(actual)
    if keys_missing_from_actual:
        errors.append(KeyError(f"Missing keys from actual: {keys_missing_from_actual}"))

    keys_missing_from_expected = set(keys_to_check) - set(expect)
    if not new_keys_in_actual_ok:
        if keys_missing_from_expected:
            errors.append(
                KeyError(f"Missing keys from expected: {keys_missing_from_expected}")
            )

    for k in keys_to_check:
        ### Check key existence
        try:
            v_expect = expect[k]
            if isinstance(v_expect, dict):
                v_expect = OrderedDict(sorted(list(v_expect.items())))
        except KeyError:
            errors.append(KeyError(f"Key {k} is missing from expected"))
            continue

        try:
            v_actual = actual[k]
            if isinstance(v_actual, dict):
                v_actual = OrderedDict(sorted(list(v_actual.items())))
        except KeyError:
            msg = f"Key {k} is missing from actual"
            errors.append(KeyError(msg))
            continue

        logging.debug(
            f"Key {repr(k)} has expected value {v_expect} and actual value {v_actual}"
        )

        ### Check NoneType special case
        if v_expect is None and v_actual is None:
            continue
        elif v_expect is None or v_actual is None:
            errors.append(
                ValueError(
                    f"Key {repr(k)}: one is None, other is not. {v_expect} vs {v_actual}"
                )
            )
            continue

        ### Check type match
        if type(v_actual) is not type(v_expect):
            errors.append(
                TypeError(
                    f"Type mismatch: type(v_actual) is not type(v_expect): {type(v_actual)} vs {type(v_expect)}"
                )
            )
            continue

        ### Check equality
        if v_expect == v_actual:
            continue
        ### This also works for strings and string arrays
        if np.array_equal(np.atleast_1d(v_expect), np.atleast_1d(v_actual)):
            continue
        ### Apply numerical tolerance
        try:
            if np.allclose(
                np.atleast_1d(v_expect),
                np.atleast_1d(v_actual),
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            ):
                continue
        except np.exceptions.DTypePromotionError:
            errors.append(
                ValueError(
                    f"Expected not equal to actual, and could not apply np.allclose. key={k}, expect={expect}, actual={actual}."
                )
            )
            continue
        except TypeError:
            if isinstance(v_expect, (dict, OrderedDict)) and isinstance(
                v_actual, (dict, OrderedDict)
            ):
                keys_not_in_v_actual = set(dict(v_expect)) - set(dict(v_actual))
                keys_in_v_expect_and_v_actual = set(dict(v_expect)) & set(
                    dict(v_actual)
                )
                keys_with_vals_not_matching = [
                    key
                    for key in keys_in_v_expect_and_v_actual
                    if v_expect[key] != v_actual[key]
                ]
                failing = []
                for key_with_vals_not_matching in keys_with_vals_not_matching:
                    if not np.allclose(
                        np.atleast_1d(v_expect[key_with_vals_not_matching]),
                        np.atleast_1d(v_actual[key_with_vals_not_matching]),
                        atol=1e-6,
                        rtol=1e-10,
                    ):
                        failing.append(
                            (
                                key_with_vals_not_matching,
                                v_expect[key_with_vals_not_matching],
                                v_actual[key_with_vals_not_matching],
                            )
                        )
                for key_with_vals_not_matching in failing:
                    errors.append(
                        ValueError(
                            f"Expected not equal to actual for key: {k}. Keys not in actual for {k}: {keys_not_in_v_actual} | keys in actual with values not matching expected for {k}: {failing}"
                        )
                    )
                continue
        errors.append(
            ValueError(
                f"Objects not equal, and numerical tolerances (atol={absolute_tolerance} rtol={relative_tolerance}) exceeded for {k}. {v_expect} vs {v_actual}."
            )
        )

    if errors:
        raise ExpectVsActualError(errors)


def rand_str(length: int) -> str:
    """Build and return a random string of length up to 32.
    Note that if this is called by different MPI ranks, each rank will receive a different random string.
    The string is not broadcasted."""
    if not (0 < length <= 32):
        raise ValueError(
            f"length requested was {length}, but this function only supports length 1 through 32"
        )
    return str(uuid.uuid4()).replace("-", "")[:length]
