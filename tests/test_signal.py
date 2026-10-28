import pytest

from src.signal import Position, Signal, SPEED_OF_LIGHT


def test_position_creation_stores_coordinates():
    pos = Position(1, 2, 3)
    assert pos.x == 1
    assert pos.y == 2
    assert pos.z == 3


def test_distance_to_self_is_zero():
    pos = Position(5, -3, 2)
    assert pos.distance(pos) == 0


def test_distance_known_3_4_5_triangle():
    a = Position(0, 0, 0)
    b = Position(3, 4, 0)
    assert a.distance(b) == pytest.approx(5)


def test_distance_is_symmetric():
    a = Position(1, 2, 3)
    b = Position(4, 0, -1)
    assert a.distance(b) == pytest.approx(b.distance(a))


def test_signal_creation_defaults():
    pos = Position(0, 0, 0)
    s = Signal("s1", frequency=1e9, time0=0, timef=0.1, position=pos)
    assert s.name == "s1"
    assert s.frequency == 1e9
    assert s.time0 == 0
    assert s.timef == 0.1
    assert s.pos is pos
    assert s.amp == 1
    assert s.phase == 0


def test_signal_creation_custom_amplitude_and_phase():
    pos = Position(1, 1, 1)
    s = Signal("s2", frequency=2.4e9, time0=1.5, timef=1.6, position=pos, amplitude=2.5, phase=0.3)
    assert s.amp == 2.5
    assert s.phase == 0.3


def test_signal_creation_stores_distinct_time0_and_timef():
    pos = Position(0, 0, 0)
    s = Signal("pulse", frequency=1e9, time0=2.0, timef=5.0, position=pos)
    assert s.time0 == 2.0
    assert s.timef == 5.0


def test_transmission_to_same_position_keeps_time_and_resets_phase():
    pos = Position(0, 0, 0)
    s = Signal("source", frequency=1e9, time0=0, timef=0.1, position=pos, amplitude=1, phase=0)
    result = s.transmission(pos, "received")
    assert result.time0 == pytest.approx(s.time0)
    assert result.timef == pytest.approx(s.timef)
    assert result.phase == 0


def test_transmission_always_resets_phase_to_zero():
    # phase is this signal's own starting phase, which is meaningless to
    # carry forward onto a different Signal object, so every transmission
    # starts fresh at 0 regardless of the originating signal's own phase.
    origin = Position(0, 0, 0)
    target = Position(3, 4, 0)
    s = Signal("source", frequency=1e9, time0=0, timef=0.1, position=origin, phase=0.7)
    result = s.transmission(target, "received")
    assert result.phase == 0


def test_transmission_sets_new_name_and_position():
    origin = Position(0, 0, 0)
    target = Position(3, 4, 0)
    s = Signal("source", frequency=1e9, time0=0, timef=0.1, position=origin)
    result = s.transmission(target, "received")
    assert result.name == "received"
    assert result.pos is target


def test_transmission_applies_transmission_factor_to_amplitude():
    origin = Position(0, 0, 0)
    target = Position(3, 4, 0)
    s = Signal("source", frequency=1e9, time0=0, timef=0.1, position=origin, amplitude=4)
    result = s.transmission(target, "received", transmission_factor=0.5)
    assert result.amp == pytest.approx(2)


def test_transmission_does_not_mutate_original_signal():
    origin = Position(0, 0, 0)
    target = Position(3, 4, 0)
    s = Signal("source", frequency=1e9, time0=0, timef=0.1, position=origin, amplitude=1)
    s.transmission(target, "received")
    assert s.pos is origin
    assert s.time0 == 0
    assert s.timef == 0.1
    assert s.name == "source"


def test_transmission_returns_signal_instance():
    origin = Position(0, 0, 0)
    target = Position(1, 1, 1)
    s = Signal("source", frequency=1e9, time0=0, timef=0.1, position=origin)
    result = s.transmission(target, "received")
    assert isinstance(result, Signal)


def test_transmission_shifts_time0_and_timef_by_the_same_travel_time():
    origin = Position(0, 0, 0)
    target = Position(3, 4, 0)  # distance = 5 m
    s = Signal("source", frequency=1e9, time0=2.0, timef=2.3, position=origin)
    travel_time = origin.distance(target) / SPEED_OF_LIGHT
    result = s.transmission(target, "received")
    assert result.time0 == pytest.approx(s.time0 + travel_time)
    assert result.timef == pytest.approx(s.timef + travel_time)


def test_transmission_preserves_pulse_duration():
    origin = Position(0, 0, 0)
    target = Position(3, 4, 0)
    s = Signal("source", frequency=1e9, time0=2.0, timef=2.3, position=origin)
    result = s.transmission(target, "received")
    original_duration = s.timef - s.time0
    result_duration = result.timef - result.time0
    assert result_duration == pytest.approx(original_duration)


def test_transmission_at_various_distances_always_resets_phase_to_zero():
    # frequency == SPEED_OF_LIGHT (Hz) makes wavelength = c / f = 1 m exactly;
    # regardless of how many (fractional) wavelengths are travelled, phase is
    # always reset to 0 on the resulting signal.
    origin = Position(0, 0, 0)
    s = Signal("source", frequency=SPEED_OF_LIGHT, time0=0, timef=0.1, position=origin, phase=0)
    for distance in (1, 1.5, 0.25):
        result = s.transmission(Position(distance, 0, 0), "received")
        assert result.phase == 0
