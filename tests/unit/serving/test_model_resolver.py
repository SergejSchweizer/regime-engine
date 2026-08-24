from __future__ import annotations

from datetime import UTC, datetime
from threading import Event, Lock, Thread

import pytest

from market_regime_engine.mlflow_support.ports import ResolvedModelVersion
from market_regime_engine.models.artifacts import GaussianHMMArtifact
from market_regime_engine.models.production_artifact import ProductionModelArtifact
from market_regime_engine.preprocessing.scaling import StandardScalerArtifact
from market_regime_engine.serving.model_cache import ModelCache, ModelCacheCapacityError
from market_regime_engine.serving.model_resolver import ModelResolver
from market_regime_engine.serving.profile_registry import ProfileModelTarget, ProfileRegistry


def artifact(*, build: str = "build-1") -> ProductionModelArtifact:
    features = ("f0",)
    return ProductionModelArtifact(
        profile_id="xetra",
        profile_config_version=1,
        registered_model="regime-xetra",
        candidate_id="gaussian_hmm_k2_full",
        state_count=2,
        source_build_id=build,
        source_data_sha256="d" * 64,
        source_schema_version=1,
        source_feature_version=1,
        data_time_semantics="current_vintage_observation_day",
        feature_selection_definition_hash="a" * 64,
        feature_selection_execution_hash="b" * 64,
        evaluation_plan_hash="c" * 64,
        evaluation_cutoff=datetime(2026, 8, 20, tzinfo=UTC),
        feature_order=features,
        scaler=StandardScalerArtifact(features, (0.0,), (1.0,), (1.0,)),
        hmm=GaussianHMMArtifact(
            state_count=2,
            feature_order=features,
            start_probabilities=(0.5, 0.5),
            transition_matrix=((0.9, 0.1), (0.1, 0.9)),
            means=((-1.0,), (1.0,)),
            full_covariances=(((1.0,),), ((1.5,),)),
        ),
        winning_seed=11,
        inference_origin_timestamp=datetime(2020, 1, 1, tzinfo=UTC),
        trained_through_timestamp=datetime(2026, 8, 19, tzinfo=UTC),
        terminal_filtered_probabilities=(0.4, 0.6),
        retained_observation_count=1500,
        skipped_incomplete_observation_count=0,
    )


class FakeRegistry:
    def __init__(self) -> None:
        self.alias_version = "1"
        self.alias_calls = 0
        self.package_calls: list[str] = []

    def resolve_alias(self, model_name: str, alias: str) -> ResolvedModelVersion:
        self.alias_calls += 1
        return ResolvedModelVersion(
            model_name=model_name,
            alias=alias,
            exact_version=self.alias_version,
            resolved_at_utc=datetime(2026, 8, 24, tzinfo=UTC),
        )

    def get_model_package_uri(self, model_name: str, exact_version: str) -> str:
        self.package_calls.append(exact_version)
        return f"file:///packages/{model_name}/{exact_version}"

    def compare_and_swap_alias(
        self,
        *,
        model_name: str,
        alias: str,
        expected_current_version: str | None,
        new_version: str,
        reason: str,
    ) -> bool:
        del model_name, alias, expected_current_version, new_version, reason
        raise AssertionError("resolver never mutates aliases")


class Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def test_profile_registry_is_data_driven_and_version_checked() -> None:
    registry = ProfileRegistry()
    xetra = registry.resolve("xetra", 1)
    assert (xetra.model_name, xetra.production_alias) == ("regime-xetra", "champion")
    with pytest.raises(KeyError, match="unknown public profile"):
        registry.resolve("crypto")
    with pytest.raises(ValueError, match="unsupported profile configuration"):
        registry.resolve("xetra", 2)

    future = ProfileRegistry(
        (
            xetra,
            ProfileModelTarget("crypto", 1, "regime-crypto", "champion"),
        )
    )
    assert future.resolve("crypto").model_name == "regime-crypto"
    with pytest.raises(ValueError, match="duplicate"):
        ProfileRegistry((xetra, xetra))
    with pytest.raises(ValueError, match="cannot be empty"):
        ProfileRegistry(())


def test_model_cache_lru_and_ref_count_safety() -> None:
    cache = ModelCache()
    first = cache.acquire("m:1", artifact)
    second = cache.acquire("m:2", lambda: artifact(build="build-2"))
    assert cache.snapshot() == (("m:1", 1), ("m:2", 1))
    with pytest.raises(ModelCacheCapacityError, match="currently referenced"):
        cache.acquire("m:3", lambda: artifact(build="build-3"))

    first.release()
    third = cache.acquire("m:3", lambda: artifact(build="build-3"))
    assert cache.snapshot() == (("m:2", 1), ("m:3", 1))
    first.release()
    second.release()
    third.release()
    with pytest.raises(ValueError, match="exactly 2"):
        ModelCache(max_versions=3)
    with pytest.raises(ValueError, match="cache key"):
        cache.acquire("", artifact)


def test_model_cache_single_flight_loads_once() -> None:
    cache = ModelCache()
    started = Event()
    proceed = Event()
    count_lock = Lock()
    load_count = 0
    leases: list[object] = []
    failures: list[BaseException] = []

    def loader() -> ProductionModelArtifact:
        nonlocal load_count
        with count_lock:
            load_count += 1
        started.set()
        assert proceed.wait(timeout=5)
        return artifact()

    def worker() -> None:
        try:
            leases.append(cache.acquire("m:1", loader))
        except BaseException as exc:
            failures.append(exc)

    first = Thread(target=worker)
    second = Thread(target=worker)
    first.start()
    assert started.wait(timeout=5)
    second.start()
    proceed.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not failures
    assert load_count == 1
    assert cache.snapshot() == (("m:1", 2),)
    for lease in leases:
        lease.release()  # type: ignore[attr-defined]


def test_resolver_ttl_explicit_bypass_and_alias_refresh() -> None:
    registry = FakeRegistry()
    clock = Clock()
    artifacts = {
        "1": artifact(build="build-1"),
        "2": artifact(build="build-2"),
        "7": artifact(build="build-7"),
    }

    def loader(uri: str) -> ProductionModelArtifact:
        return artifacts[uri.rsplit("/", 1)[-1]]

    resolver = ModelResolver(registry, alias_ttl_seconds=10.0, package_loader=loader, clock=clock)
    with resolver.resolve("xetra") as resolved:
        assert resolved.exact_version == "1"
        assert resolved.resolved_via_alias == "champion"
        assert resolved.artifact.source_build_id == "build-1"
    assert registry.alias_calls == 1

    registry.alias_version = "2"
    with resolver.resolve("xetra") as cached_alias:
        assert cached_alias.exact_version == "1"
    assert registry.alias_calls == 1

    clock.value = 111.0
    with resolver.resolve("xetra") as refreshed:
        assert refreshed.exact_version == "2"
    assert registry.alias_calls == 2

    with resolver.resolve("xetra", exact_version="7") as explicit:
        assert explicit.exact_version == "7"
        assert explicit.resolved_via_alias is None
    assert registry.alias_calls == 2
    with pytest.raises(ValueError, match="explicit model version"):
        resolver.resolve("xetra", exact_version="")


def test_invalid_new_champion_fails_without_falling_back_or_relabelling() -> None:
    registry = FakeRegistry()
    clock = Clock()

    def loader(uri: str) -> ProductionModelArtifact:
        version = uri.rsplit("/", 1)[-1]
        if version == "2":
            raise ValueError("invalid new champion package")
        return artifact(build="good")

    resolver = ModelResolver(registry, alias_ttl_seconds=10.0, package_loader=loader, clock=clock)
    with resolver.resolve("xetra") as initial:
        assert initial.exact_version == "1"

    registry.alias_version = "2"
    clock.value = 111.0
    with pytest.raises(ValueError, match="invalid new champion package"):
        resolver.resolve("xetra")
    assert registry.package_calls[-1] == "2"

    with pytest.raises(ValueError, match="invalid new champion package"):
        resolver.resolve("xetra")
    assert registry.alias_calls == 2
    assert registry.package_calls[-1] == "2"

    with resolver.resolve("xetra", exact_version="1") as old_exact:
        assert old_exact.exact_version == "1"
        assert old_exact.artifact.source_build_id == "good"


def test_resolver_validates_alias_identity_and_configuration() -> None:
    class BadRegistry(FakeRegistry):
        def resolve_alias(self, model_name: str, alias: str) -> ResolvedModelVersion:
            return ResolvedModelVersion(
                model_name="other",
                alias=alias,
                exact_version="1",
                resolved_at_utc=datetime(2026, 8, 24, tzinfo=UTC),
            )

    with pytest.raises(ValueError, match="mismatched model identity"):
        ModelResolver(BadRegistry(), package_loader=lambda _: artifact()).resolve("xetra")
    with pytest.raises(ValueError, match="alias TTL"):
        ModelResolver(FakeRegistry(), alias_ttl_seconds=0.0)

    def wrong_version_loader(_: str) -> ProductionModelArtifact:
        return artifact()

    custom = ProfileRegistry((ProfileModelTarget("xetra", 2, "regime-xetra", "champion"),))
    with pytest.raises(ValueError, match="configuration version differs"):
        ModelResolver(
            FakeRegistry(),
            profiles=custom,
            package_loader=wrong_version_loader,
        ).resolve("xetra")
