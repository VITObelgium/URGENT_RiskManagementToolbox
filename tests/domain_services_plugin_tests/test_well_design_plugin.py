"""Unit tests for the built-in well_design domain service plugin.

Covers: request-model validation, MD sampling, plane/3-D geometry of all four
well templates, perforation handling on both the inbound (validation) and
outbound (completion) paths, and the service/payload contract.
"""

import math

import pytest
from pydantic import ValidationError

# Adjust this import to the plugin's real module path in your repo.
from plugins.domain_services.well_design import (
    H_WELL_DLS,
    MD_ATOL,
    HWellModel,
    IWellModel,
    JWellModel,
    SWellModel,
    WellDesignDomainService,
    WellDesignServiceRequest,
    WellDesignServiceResponse,
    _h_well_section_mds,
    _plane_positions,
    _sample_mds,
    plugin,
)

# Radius of the fixed H-well build curve (90 deg at 4 deg / 30 m).
H_CURVE_RADIUS = 2.0 * (90.0 * 30.0 / H_WELL_DLS) / math.pi

WELLHEAD = {"x": 0.0, "y": 0.0, "z": 0.0}


def iwell(**overrides):
    payload = {
        "well_type": "IWell",
        "name": "w",
        "wellhead": WELLHEAD,
        "md": 10.0,
        "md_step": 1.0,
    }
    payload.update(overrides)
    return IWellModel(**payload)


def jwell(**overrides):
    payload = {
        "well_type": "JWell",
        "name": "w",
        "wellhead": WELLHEAD,
        "md_linear1": 100.0,
        "md_curved": 675.0,  # 90 deg at dls=4
        "dls": 4.0,
        "md_linear2": 200.0,
        "azimuth": 0.0,
        "md_step": 0.5,
    }
    payload.update(overrides)
    return JWellModel(**payload)


def build(model):
    """Build a single well through the public service path."""
    service = WellDesignDomainService()
    payload = service.build_payload([model.model_dump()])
    return payload["wells"][0]


def polyline_length(trajectory):
    return sum(
        math.dist(a, b) for a, b in zip(trajectory, trajectory[1:], strict=False)
    )


# ---------------------------------------------------------------------------
# Request-model validation
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_perforation_range_must_be_increasing(self):
        with pytest.raises(ValidationError, match="must be less than"):
            iwell(md=100.0, perforations={"p": {"start_md": 20.0, "end_md": 20.0}})

    def test_perforation_range_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            iwell(md=100.0, perforations={"p": {"start_md": -1.0, "end_md": 5.0}})

    def test_overlapping_perforations_rejected(self):
        with pytest.raises(ValidationError, match="overlap"):
            iwell(
                md=100.0,
                perforations={
                    "a": {"start_md": 0.0, "end_md": 15.0},
                    "b": {"start_md": 10.0, "end_md": 20.0},
                },
            )

    def test_touching_perforations_allowed(self):
        model = iwell(
            md=100.0,
            perforations={
                "a": {"start_md": 0.0, "end_md": 10.0},
                "b": {"start_md": 10.0, "end_md": 20.0},
            },
        )
        assert set(model.perforations) == {"a", "b"}

    def test_perforations_sorted_by_start_md(self):
        model = iwell(
            md=100.0,
            perforations={
                "late": {"start_md": 50.0, "end_md": 60.0},
                "early": {"start_md": 5.0, "end_md": 10.0},
            },
        )
        assert list(model.perforations) == ["early", "late"]

    def test_perforation_clipped_to_well_length(self):
        model = iwell(md=100.0, perforations={"p": {"start_md": 90.0, "end_md": 150.0}})
        assert model.perforations["p"].end_md == pytest.approx(100.0)

    def test_perforation_fully_beyond_well_dropped(self):
        model = iwell(
            md=100.0, perforations={"p": {"start_md": 100.0, "end_md": 150.0}}
        )
        assert model.perforations is None

    def test_md_step_lower_bound(self):
        with pytest.raises(ValidationError):
            iwell(md_step=0.05)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            iwell(unexpected=1)

    def test_azimuth_bounds(self):
        with pytest.raises(ValidationError):
            jwell(azimuth=360.0)

    def test_dls_bounds(self):
        with pytest.raises(ValidationError):
            jwell(dls=45.01)
        with pytest.raises(ValidationError):
            jwell(dls=-45.0)  # gt=-45 is exclusive

    def test_hwell_depth_smaller_than_curve_radius_rejected(self):
        with pytest.raises(ValidationError, match="Increase depth"):
            HWellModel(
                name="h",
                wellhead=WELLHEAD,
                TVD=H_CURVE_RADIUS - 1.0,
                md_lateral=1000.0,
                azimuth=0.0,
            )

    def test_hwell_lateral_smaller_than_curve_radius_rejected(self):
        with pytest.raises(ValidationError, match="Increase width"):
            HWellModel(
                name="h",
                wellhead=WELLHEAD,
                TVD=1000.0,
                md_lateral=H_CURVE_RADIUS - 1.0,
                azimuth=0.0,
            )

    def test_discriminated_union_dispatch(self):
        adapter = WellDesignDomainService.get_item_state_adapter()
        model = adapter.validate_python(
            {"well_type": "IWell", "name": "w", "wellhead": WELLHEAD, "md": 10.0}
        )
        assert isinstance(model, IWellModel)

    def test_unknown_well_type_rejected(self):
        adapter = WellDesignDomainService.get_item_state_adapter()
        with pytest.raises(ValidationError):
            adapter.validate_python(
                {"well_type": "ZWell", "name": "w", "wellhead": WELLHEAD, "md": 10.0}
            )

    def test_duplicate_well_names_rejected(self):
        models = [iwell(name="dup").model_dump(), iwell(name="dup").model_dump()]
        with pytest.raises(ValidationError, match="unique"):
            WellDesignServiceRequest(models=models)


# ---------------------------------------------------------------------------
# MD sampling
# ---------------------------------------------------------------------------


class TestSampleMds:
    def test_regular_grid_with_exact_end(self):
        assert _sample_mds([(10.0, 0.0)], 1.0, []) == pytest.approx(
            [float(i) for i in range(11)]
        )

    def test_remainder_appended(self):
        mds = _sample_mds([(10.3, 0.0)], 1.0, [])
        assert mds[-1] == pytest.approx(10.3)
        assert mds[-2] == pytest.approx(10.0)

    def test_grid_restarts_at_section_boundary(self):
        mds = _sample_mds([(10.3, 0.0), (5.0, 4.0)], 1.0, [])
        # Boundary itself is a sample point, and the grid restarts from it.
        assert any(abs(m - 10.3) < MD_ATOL for m in mds)
        assert any(abs(m - 11.3) < MD_ATOL for m in mds)
        assert mds[-1] == pytest.approx(15.3)

    def test_extra_mds_merged_and_out_of_range_ignored(self):
        mds = _sample_mds([(10.0, 0.0)], 1.0, [2.5, 7.75, -1.0, 11.0])
        assert any(abs(m - 2.5) < MD_ATOL for m in mds)
        assert any(abs(m - 7.75) < MD_ATOL for m in mds)
        assert all(0.0 <= m <= 10.0 for m in mds)

    def test_near_duplicates_deduplicated(self):
        mds = _sample_mds([(10.0, 0.0)], 1.0, [7.0 + MD_ATOL / 2])
        assert len([m for m in mds if abs(m - 7.0) < 2 * MD_ATOL]) == 1

    def test_strictly_increasing(self):
        mds = _sample_mds([(10.3, 0.0), (675.0, 4.0), (5.5, 0.0)], 0.5, [3.14, 700.0])
        assert all(b - a > MD_ATOL for a, b in zip(mds, mds[1:], strict=False))


# ---------------------------------------------------------------------------
# Geometry: straight sections
# ---------------------------------------------------------------------------


class TestVerticalGeometry:
    def test_vertical_well_points(self):
        well = build(
            iwell(md=10.0, md_step=1.0, wellhead={"x": 5.0, "y": 6.0, "z": 7.0})
        )
        trajectory = well["trajectory"]
        assert len(trajectory) == 11
        assert trajectory[0] == pytest.approx((5.0, 6.0, 7.0))
        assert trajectory[-1] == pytest.approx((5.0, 6.0, 17.0))
        xs, ys, zs = zip(*trajectory, strict=False)
        assert all(x == pytest.approx(5.0) for x in xs)
        assert all(y == pytest.approx(6.0) for y in ys)
        assert list(zs) == pytest.approx([7.0 + i for i in range(11)])

    def test_remainder_point_reaches_exact_total_md(self):
        well = build(iwell(md=10.3, md_step=1.0))
        assert well["trajectory"][-1][2] == pytest.approx(10.3)

    def test_azimuth_has_no_effect_on_vertical_iwell(self):
        # IWell has no azimuth field at all; x/y must stay at the wellhead.
        well = build(iwell(md=50.0))
        assert all(p[0] == pytest.approx(0.0) for p in well["trajectory"])
        assert all(p[1] == pytest.approx(0.0) for p in well["trajectory"])


# ---------------------------------------------------------------------------
# Geometry: curved sections
# ---------------------------------------------------------------------------


class TestCurvedGeometry:
    def test_jwell_quarter_circle_endpoints(self):
        # 675 m at 4 deg/30 m turns exactly 90 deg with radius 1350/pi.
        well = build(jwell())
        r = H_CURVE_RADIUS
        assert well["trajectory"][-1] == pytest.approx((r + 200.0, 0.0, 100.0 + r))

    def test_jwell_lateral_leg_is_horizontal(self):
        well = build(jwell())
        last, second_last = well["trajectory"][-1], well["trajectory"][-2]
        assert last[2] == pytest.approx(second_last[2])  # z constant
        assert last[0] > second_last[0]  # marching in +x

    def test_negative_dls_curves_to_negative_x(self):
        well = build(jwell(dls=-4.0))
        assert well["trajectory"][-1][0] == pytest.approx(-(H_CURVE_RADIUS + 200.0))

    def test_zero_dls_degrades_to_straight_well(self):
        well = build(jwell(dls=0.0))
        total_md = 100.0 + 675.0 + 200.0
        assert well["trajectory"][-1] == pytest.approx((0.0, 0.0, total_md))

    def test_azimuth_rotates_lateral_into_y(self):
        well = build(jwell(azimuth=90.0))
        x, y, z = well["trajectory"][-1]
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(H_CURVE_RADIUS + 200.0)
        assert z == pytest.approx(100.0 + H_CURVE_RADIUS)

    def test_wellhead_translation_applied_after_rotation(self):
        well = build(jwell(azimuth=90.0, wellhead={"x": 10.0, "y": 20.0, "z": 30.0}))
        x, y, z = well["trajectory"][-1]
        assert x == pytest.approx(10.0, abs=1e-9)
        assert y == pytest.approx(20.0 + H_CURVE_RADIUS + 200.0)
        assert z == pytest.approx(30.0 + 100.0 + H_CURVE_RADIUS)

    def test_polyline_length_matches_measured_depth(self):
        well = build(jwell())
        total_md = 100.0 + 675.0 + 200.0
        assert polyline_length(well["trajectory"]) == pytest.approx(total_md, rel=1e-4)

    def test_consecutive_points_never_farther_apart_than_step(self):
        step = 0.5
        well = build(jwell(md_step=step))
        trajectory = well["trajectory"]
        assert all(
            math.dist(a, b) <= step + MD_ATOL
            for a, b in zip(trajectory, trajectory[1:], strict=False)
        )

    def test_swell_returns_to_vertical(self):
        # +50 deg over curve 1, -50 deg over curve 2 -> final leg vertical.
        model = SWellModel(
            name="s",
            wellhead=WELLHEAD,
            md_linear1=100.0,
            md_curved1=150.0,
            dls1=10.0,
            md_linear2=100.0,
            md_curved2=150.0,
            dls2=-10.0,
            md_linear3=100.0,
            azimuth=0.0,
        )
        well = build(model)
        last, second_last = well["trajectory"][-1], well["trajectory"][-2]
        assert last[0] == pytest.approx(second_last[0])  # x frozen again
        assert last[2] - second_last[2] == pytest.approx(0.5)  # pure z advance
        assert last[0] > 0.0  # the S-shape displaced the well laterally

    def test_hwell_lands_at_tvd_and_lateral_reach(self):
        model = HWellModel(
            name="h",
            wellhead=WELLHEAD,
            TVD=1000.0,
            md_lateral=800.0,
            azimuth=0.0,
        )
        well = build(model)
        assert well["trajectory"][-1] == pytest.approx((800.0, 0.0, 1000.0))
        # Heel of the lateral (end of the build curve) sits exactly at TVD.
        heel_zs = [p[2] for p in well["trajectory"] if p[0] >= H_CURVE_RADIUS - MD_ATOL]
        assert all(z == pytest.approx(1000.0, abs=1e-6) for z in heel_zs)

    def test_hwell_section_split_consistent_with_total(self):
        vertical, curved, horizontal = _h_well_section_mds(1000.0, 800.0)
        assert vertical == pytest.approx(1000.0 - H_CURVE_RADIUS)
        assert curved == pytest.approx(675.0)
        assert horizontal == pytest.approx(800.0 - H_CURVE_RADIUS)

    def test_plane_positions_tangent_continuity(self):
        # No kink at the section joints: direction change across a boundary
        # stays within what the arc curvature allows for one step.
        sections = [(100.0, 0.0), (300.0, 8.0), (100.0, 0.0)]
        mds = _sample_mds(sections, 1.0, [])
        pts = _plane_positions(sections, mds)
        max_turn = math.radians(8.0) / 30.0 * 1.0  # rad per 1 m step
        for (x0, z0, _), (x1, z1, _), (x2, z2, _) in zip(
            pts, pts[1:], pts[2:], strict=False
        ):
            a = math.atan2(x1 - x0, z1 - z0)
            b = math.atan2(x2 - x1, z2 - z1)
            assert abs(b - a) <= max_turn + 1e-6


# ---------------------------------------------------------------------------
# Perforations on the outbound path
# ---------------------------------------------------------------------------


class TestCompletionOutput:
    def test_no_perforations_means_no_completion(self):
        assert build(iwell())["completion"] is None

    def test_perforation_boundaries_become_exact_trajectory_points(self):
        well = build(
            iwell(
                md=100.0,
                md_step=7.0,
                perforations={"p": {"start_md": 10.5, "end_md": 20.25}},
            )
        )
        zs = [p[2] for p in well["trajectory"]]
        assert any(abs(z - 10.5) < MD_ATOL for z in zs)
        assert any(abs(z - 20.25) < MD_ATOL for z in zs)

    def test_perforation_points_cover_exactly_the_range(self):
        well = build(
            iwell(
                md=100.0,
                md_step=7.0,
                perforations={"p": {"start_md": 10.5, "end_md": 20.25}},
            )
        )
        (perf,) = well["completion"]["perforations"]
        assert perf["name"] == "p"
        assert perf["range"] == pytest.approx((10.5, 20.25))
        zs = [p[2] for p in perf["points"]]
        # Grid inside the range is 14.0 only; boundaries added -> 3 points.
        assert zs == pytest.approx([10.5, 14.0, 20.25])
        assert all(10.5 - MD_ATOL <= z <= 20.25 + MD_ATOL for z in zs)

    def test_multiple_perforations_reported_in_md_order(self):
        well = build(
            iwell(
                md=100.0,
                perforations={
                    "deep": {"start_md": 50.0, "end_md": 60.0},
                    "shallow": {"start_md": 5.0, "end_md": 10.0},
                },
            )
        )
        names = [p["name"] for p in well["completion"]["perforations"]]
        assert names == ["shallow", "deep"]

    def test_clipped_perforation_range_survives_to_output(self):
        well = build(
            iwell(md=100.0, perforations={"p": {"start_md": 90.0, "end_md": 150.0}})
        )
        (perf,) = well["completion"]["perforations"]
        assert perf["range"] == pytest.approx((90.0, 100.0))
        assert perf["points"][-1][2] == pytest.approx(100.0)

    def test_perforation_spanning_a_curve_lies_on_the_arc(self):
        # Regression for the original curved-section perforation bug: every
        # perforation point inside the curve must sit at radius R from the
        # arc's rotation centre.
        model = jwell(perforations={"p": {"start_md": 150.0, "end_md": 300.0}})
        well = build(model)
        (perf,) = well["completion"]["perforations"]
        r = H_CURVE_RADIUS
        center_x, center_z = r, 100.0  # curve starts at (0, 100), builds toward +x
        for x, _, z in perf["points"]:
            radius = math.hypot(x - center_x, z - center_z)
            assert radius == pytest.approx(r, abs=1e-6)


# ---------------------------------------------------------------------------
# Service / payload contract
# ---------------------------------------------------------------------------


class TestService:
    def test_payload_shape_round_trips_through_response_model(self):
        service = WellDesignDomainService()
        payload = service.build_payload(
            [iwell(name="a").model_dump(), jwell(name="b").model_dump()]
        )
        assert set(payload) == {"wells"}
        assert [w["name"] for w in payload["wells"]] == ["a", "b"]
        WellDesignServiceResponse(**payload)  # payload is re-validatable

    def test_build_payload_accepts_plain_dict_items(self):
        service = WellDesignDomainService()
        payload = service.build_payload(
            [{"well_type": "IWell", "name": "w", "wellhead": WELLHEAD, "md": 5.0}]
        )
        assert payload["wells"][0]["trajectory"][-1][2] == pytest.approx(5.0)

    def test_build_payload_rejects_duplicate_names(self):
        service = WellDesignDomainService()
        with pytest.raises(ValidationError, match="unique"):
            service.build_payload(
                [iwell(name="dup").model_dump(), iwell(name="dup").model_dump()]
            )

    def test_build_payload_rejects_invalid_item(self):
        service = WellDesignDomainService()
        with pytest.raises(ValidationError):
            service.build_payload([{"well_type": "IWell", "name": "w"}])

    def test_plugin_descriptor(self):
        assert plugin.name == "well_design"
        assert plugin.implementation is WellDesignDomainService
        assert WellDesignDomainService.ServiceName == "well_design"
