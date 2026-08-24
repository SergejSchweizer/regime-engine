"""Data-driven public-profile to registered-model routing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProfileModelTarget:
    profile_id: str
    profile_config_version: int
    model_name: str
    production_alias: str

    def __post_init__(self) -> None:
        if not self.profile_id or not self.model_name or not self.production_alias:
            raise ValueError("profile model target identity fields cannot be empty")
        if self.profile_config_version < 1:
            raise ValueError("profile config version must be positive")


DEFAULT_PROFILE_TARGETS = (
    ProfileModelTarget(
        profile_id="xetra",
        profile_config_version=1,
        model_name="regime-xetra",
        production_alias="champion",
    ),
)


class ProfileRegistry:
    def __init__(self, targets: tuple[ProfileModelTarget, ...] = DEFAULT_PROFILE_TARGETS) -> None:
        if not targets:
            raise ValueError("profile registry cannot be empty")
        mapping = {target.profile_id: target for target in targets}
        if len(mapping) != len(targets):
            raise ValueError("profile registry contains duplicate profile IDs")
        self._targets = mapping

    def resolve(
        self,
        profile_id: str,
        profile_config_version: int | None = None,
    ) -> ProfileModelTarget:
        try:
            target = self._targets[profile_id]
        except KeyError as exc:
            raise KeyError(f"unknown public profile: {profile_id}") from exc
        if (
            profile_config_version is not None
            and profile_config_version != target.profile_config_version
        ):
            raise ValueError(
                f"unsupported profile configuration version for {profile_id}: "
                f"{profile_config_version}"
            )
        return target

    def targets(self) -> tuple[ProfileModelTarget, ...]:
        return tuple(self._targets.values())
