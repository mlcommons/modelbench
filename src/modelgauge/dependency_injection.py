from modelgauge.general import get_class
from modelgauge.secret_values import (
    Injector,
    MissingSecretValues,
    RawSecrets,
    Secret,
    SerializedSecret,
)
from typing import Any, Dict, Mapping, Sequence, Tuple


def inject_dependencies(
    args: Sequence[Any], kwargs: Mapping[str, Any], secrets: RawSecrets
) -> Tuple[Sequence[Any], Mapping[str, Any]]:
    """Replace any arg or kwarg injectors with their concrete values."""
    replaced_args = []
    missing_secrets = []
    for arg in args:
        try:
            replaced_args.append(_replace_with_injected(arg, secrets))
        except MissingSecretValues as e:
            missing_secrets.append(e)
        # TODO Catch other kinds of missing dependencies

    replaced_kwargs: Dict[str, Any] = {}
    for key, arg in kwargs.items():
        try:
            replaced_kwargs[key] = _replace_with_injected(arg, secrets)
        except MissingSecretValues as e:
            missing_secrets.append(e)
        # TODO Catch other kinds of missing dependencies
    if missing_secrets:
        raise MissingSecretValues.combine(missing_secrets)

    return replaced_args, replaced_kwargs


def list_dependency_usage(
    args: Sequence[Any], kwargs: Mapping[str, Any], secrets: RawSecrets
) -> Tuple[Sequence[Any], Sequence[Any]]:
    """List all secrets used in the given args and kwargs."""

    def process_item(item):
        """Process an individual item (arg or kwarg)."""
        try:
            replaced_item = _replace_with_injected(item, secrets)
            if isinstance(item, (Injector, SerializedSecret)):
                used_dependencies.append(replaced_item)
        except MissingSecretValues as e:
            missing_dependencies.extend(
                [
                    {
                        "scope": desc.scope,
                        "key": desc.key,
                        "instructions": desc.instructions,
                    }
                    for desc in e.descriptions
                ]
            )
        # TODO Catch other kinds of missing dependencies

    used_dependencies: Sequence[Any] = []
    missing_dependencies: Sequence[Any] = []
    # optional_dependencies: Sequence[Any] = []

    for item in list(args) + list(kwargs.values()):
        process_item(item)

    return used_dependencies, missing_dependencies


def _replace_with_injected(value, secrets: RawSecrets):
    if isinstance(value, Injector):
        return value.inject(secrets)
    if isinstance(value, SerializedSecret):
        cls = get_class(value.module, value.class_name)
        assert issubclass(cls, Secret)
        return cls.make(secrets)
    return value


def serialize_injected_dependencies(
    args: Sequence[Any], kwargs: Mapping[str, Any]
) -> Tuple[Sequence[Any], Mapping[str, Any]]:
    """Replace any injected values with their safe-to-serialize form."""
    replaced_args = []
    for arg in args:
        replaced_args.append(_serialize(arg))
    replaced_kwargs: Dict[str, Any] = {}
    for key, arg in kwargs.items():
        replaced_kwargs[key] = _serialize(arg)
    return replaced_args, replaced_kwargs


def _serialize(arg):
    # TODO Try to make this more generic.
    if isinstance(arg, Secret):
        return SerializedSecret.serialize(arg)
    return arg
