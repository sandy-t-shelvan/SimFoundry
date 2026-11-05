"""MessyKitchens scene scores: mean the seeds of each object, then the objects.

A rectangular seed table makes that reduction equal to a pooled mean and to a
mean taken across objects inside each seed. The order test uses unequal seed
counts, which is the case those three numbers come apart. ``expected_seeds``
is None there because the listed seeds are the complete set, not a hole in
``(0, 1, 2)``. The missing-seed test keeps the default and must raise.
"""

import pytest

from sceneagent.metrics import MissingSeedError, aggregate_messy_kitchens


def _record(f1, chamfer_m):
    return {"f1": f1, "chamfer_m": chamfer_m}


def _constant_seeds(f1, chamfer_m, n_seeds=3):
    return {seed: _record(f1, chamfer_m) for seed in range(n_seeds)}


def test_seed_then_object_order_differs_from_the_other_reductions():
    cup_f1 = (0.0, 0.0)
    bowl_f1 = (1.0,)
    cup_chamfer = (0.02, 0.04)
    bowl_chamfer = (0.09,)
    per_object = {
        "cup": {
            0: _record(cup_f1[0], cup_chamfer[0]),
            1: _record(cup_f1[1], cup_chamfer[1]),
        },
        "bowl": {
            0: _record(bowl_f1[0], bowl_chamfer[0]),
        },
    }
    result = aggregate_messy_kitchens(
        per_object,
        expected_objects=("cup", "bowl"),
        expected_seeds=None,
    )

    seed_then_object_f1 = ((cup_f1[0] + cup_f1[1]) / 2 + bowl_f1[0]) / 2
    pooled_f1 = (cup_f1[0] + cup_f1[1] + bowl_f1[0]) / 3
    seed_major_f1 = (((cup_f1[0] + bowl_f1[0]) / 2) + cup_f1[1]) / 2
    assert seed_then_object_f1 == pytest.approx(0.5)
    assert pooled_f1 == pytest.approx(1.0 / 3.0)
    assert seed_major_f1 == pytest.approx(0.25)
    assert result["f1_mean"] == pytest.approx(seed_then_object_f1)
    assert result["f1_mean"] != pytest.approx(pooled_f1)
    assert result["f1_mean"] != pytest.approx(seed_major_f1)

    seed_then_object_chamfer = ((cup_chamfer[0] + cup_chamfer[1]) / 2 + bowl_chamfer[0]) / 2
    pooled_chamfer = (cup_chamfer[0] + cup_chamfer[1] + bowl_chamfer[0]) / 3
    seed_major_chamfer = (((cup_chamfer[0] + bowl_chamfer[0]) / 2) + cup_chamfer[1]) / 2
    assert result["chamfer_mean_m"] == pytest.approx(seed_then_object_chamfer)
    assert result["chamfer_mean_m"] != pytest.approx(pooled_chamfer)
    assert result["chamfer_mean_m"] != pytest.approx(seed_major_chamfer)
    assert result["per_object"]["cup"]["f1_mean"] == pytest.approx(0.0)
    assert result["per_object"]["bowl"]["f1_mean"] == pytest.approx(1.0)
    assert result["per_object"]["cup"]["chamfer_mean_m"] == pytest.approx(0.03)
    assert result["aggregation"] == "seeds_then_objects"
    assert result["scored_objects"] == ["bowl", "cup"]
    assert result["full_scene"] is False
    assert "bowl" in result["summary"]
    assert "cup" in result["summary"]
    assert "bowl seeds 0" in result["summary"]
    assert "cup seeds 0, 1" in result["summary"]


def test_three_seeds_then_equal_object_weights():
    per_object = {
        "cup": {
            0: _record(0.0, 0.01),
            1: _record(0.0, 0.03),
            2: _record(0.6, 0.05),
        },
        "plate": {
            0: _record(1.0, 0.02),
            1: _record(1.0, 0.02),
            2: _record(0.4, 0.08),
        },
    }
    result = aggregate_messy_kitchens(per_object, expected_objects=("plate", "cup"))
    assert result["per_object"]["cup"]["f1_mean"] == pytest.approx(0.2)
    assert result["per_object"]["plate"]["f1_mean"] == pytest.approx(0.8)
    assert result["f1_mean"] == pytest.approx(0.5)
    assert result["per_object"]["cup"]["chamfer_mean_m"] == pytest.approx(0.03)
    assert result["per_object"]["plate"]["chamfer_mean_m"] == pytest.approx(0.04)
    assert result["chamfer_mean_m"] == pytest.approx(0.035)
    assert result["full_scene"] is True
    assert result["objects_complete"] is True
    assert result["protocol_seeds_complete"] is True
    assert result["reported_as"] == "messy_kitchens_full_scene"
    assert result["omitted_objects"] == []
    assert result["scored_objects"] == ["cup", "plate"]
    assert result["summary"].startswith("Full MessyKitchens scene score over cup, plate.")

    unnamed = aggregate_messy_kitchens(per_object)
    assert unnamed["f1_mean"] == pytest.approx(result["f1_mean"])
    assert unnamed["full_scene"] is False
    assert unnamed["reported_as"] == "messy_kitchens_subset"
    assert unnamed["omitted_objects"] is None
    assert "cup" in unnamed["summary"]
    assert "plate" in unnamed["summary"]
    assert "No expected object list was provided." in unnamed["summary"]


def test_omitted_object_names_the_scored_subset():
    per_object = {
        "cup": _constant_seeds(0.10, 0.01),
        "plate": _constant_seeds(0.40, 0.03),
        "bowl": _constant_seeds(0.70, 0.05),
    }
    result = aggregate_messy_kitchens(
        per_object,
        expected_objects=("cup", "plate", "bowl", "knife_block"),
    )
    assert result["f1_mean"] == pytest.approx(0.40)
    assert result["chamfer_mean_m"] == pytest.approx(0.03)
    # Filling the omitted object with zero is the bin rule. It would move both means.
    assert result["f1_mean"] != pytest.approx(0.30)
    assert result["chamfer_mean_m"] != pytest.approx(0.0225)
    assert result["full_scene"] is False
    assert result["objects_complete"] is False
    assert result["reported_as"] == "messy_kitchens_subset"
    assert result["scored_objects"] == ["bowl", "cup", "plate"]
    assert result["omitted_objects"] == ["knife_block"]
    for name in ("bowl", "cup", "plate", "knife_block"):
        assert name in result["summary"]
    assert "Subset score over bowl, cup, plate." in result["summary"]
    assert "Omitted: knife_block." in result["summary"]
    assert "not the full-scene MessyKitchens score" in result["summary"]

    complete = aggregate_messy_kitchens(
        per_object,
        expected_objects=("bowl", "cup", "plate"),
    )
    assert complete["full_scene"] is True
    assert complete["f1_mean"] == pytest.approx(result["f1_mean"])
    assert "knife_block" not in complete["summary"]


def test_missing_seed_is_not_averaged():
    per_object = {
        "cup": _constant_seeds(0.2, 0.01),
        "plate": {
            0: _record(0.0, 0.02),
            2: _record(1.0, 0.04),
        },
    }
    with pytest.raises(MissingSeedError, match="plate") as caught:
        aggregate_messy_kitchens(per_object, expected_objects=("cup", "plate"))
    err = caught.value
    assert err.object_name == "plate"
    assert err.missing_seeds == (1,)
    assert "not averaged" in str(err)
    assert isinstance(err, ValueError)

    per_object["plate"][1] = {"f1": None, "chamfer_m": 0.04}
    with pytest.raises(MissingSeedError, match="seed 1") as caught:
        aggregate_messy_kitchens(per_object, expected_objects=("cup", "plate"))
    assert "not averaged" in str(caught.value)

    per_object["plate"][1] = {"f1": 0.5, "chamfer_m": float("nan")}
    with pytest.raises(MissingSeedError, match="plate") as caught:
        aggregate_messy_kitchens(per_object, expected_objects=("cup", "plate"))
    assert caught.value.missing_seeds == (1,)
    assert "not averaged" in str(caught.value)

    # expected_seeds=None averages listed seeds. An explicit hole is still a missing seed.
    with pytest.raises(MissingSeedError, match="cup") as caught:
        aggregate_messy_kitchens(
            {
                "cup": {0: _record(0.0, 0.02), 1: {"f1": None, "chamfer_m": 0.04}},
                "bowl": {0: _record(1.0, 0.09)},
            },
            expected_objects=("cup", "bowl"),
            expected_seeds=None,
        )
    assert caught.value.missing_seeds == (1,)
    assert "not averaged" in str(caught.value)

    with pytest.raises(MissingSeedError, match="cup") as caught:
        aggregate_messy_kitchens({"cup": {}}, expected_objects=("cup",))
    assert caught.value.missing_seeds == (0, 1, 2)


def test_empty_input():
    with pytest.raises(ValueError, match="empty input") as caught:
        aggregate_messy_kitchens({})
    assert not isinstance(caught.value, MissingSeedError)

    with pytest.raises(ValueError, match="empty input") as caught:
        aggregate_messy_kitchens({}, expected_objects=("cup", "knife_block"))
    assert not isinstance(caught.value, MissingSeedError)
    message = str(caught.value)
    assert "cup" in message
    assert "knife_block" in message

    with pytest.raises(ValueError, match="empty input"):
        aggregate_messy_kitchens({}, expected_objects=())


def test_object_outside_the_expected_scene_is_rejected():
    with pytest.raises(ValueError, match="spoon"):
        aggregate_messy_kitchens(
            {
                "cup": _constant_seeds(0.2, 0.01),
                "spoon": _constant_seeds(0.4, 0.02),
            },
            expected_objects=("cup",),
        )
