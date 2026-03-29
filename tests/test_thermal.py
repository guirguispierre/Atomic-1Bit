import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import torch

from atomic_1bit.utils.thermal import ThermalMonitor


# On macOS, psutil.sensors_temperatures doesn't exist, so we need create=True
PATCH_TARGET = "psutil.sensors_temperatures"


class TestThermalMonitorInit:
    def test_default_thresholds(self):
        """Check default threshold values."""
        with patch(PATCH_TARGET, create=True, return_value={}):
            monitor = ThermalMonitor()
        assert monitor.high == 80.0
        assert monitor.resume == 70.0
        assert monitor.interval == 50

    def test_custom_thresholds(self):
        with patch(PATCH_TARGET, create=True, return_value={}):
            monitor = ThermalMonitor(high_threshold=90.0, resume_threshold=75.0, check_interval_steps=100)
        assert monitor.high == 90.0
        assert monitor.resume == 75.0
        assert monitor.interval == 100

    def test_disabled_when_no_sensors(self):
        """Should disable gracefully when no sensors are found."""
        with patch(PATCH_TARGET, create=True, return_value={}):
            monitor = ThermalMonitor()
        assert not monitor.enabled

    def test_disabled_when_no_sensor_api(self):
        """Should disable gracefully when sensors_temperatures doesn't exist."""
        # ThermalMonitor checks hasattr(psutil, "sensors_temperatures")
        # On macOS this is already False, so just construct directly
        monitor = ThermalMonitor()
        assert not monitor.enabled


class TestThermalMonitorCheck:
    def test_skip_when_disabled(self):
        """check_and_pause should be a no-op when disabled."""
        monitor = ThermalMonitor()
        assert not monitor.enabled
        # Should not raise or block
        monitor.check_and_pause(step=0)

    def test_skip_non_interval_steps(self):
        """Should only check on interval steps."""
        monitor = ThermalMonitor(check_interval_steps=10)
        monitor.enabled = True
        # Step 5 is not a multiple of 10, should skip
        with patch.object(monitor, '_get_max_temp') as mock_get:
            monitor.check_and_pause(step=5)
            mock_get.assert_not_called()

    def test_get_max_temp_returns_float_or_none(self):
        """_get_max_temp should return a float or None when sensors unavailable."""
        monitor = ThermalMonitor()
        temp = monitor._get_max_temp()
        assert temp is None or isinstance(temp, float)

    def test_max_temp_extraction(self):
        """Should return the maximum temperature across all sensors."""
        mock_temps = {
            "cpu": [MagicMock(current=65.0), MagicMock(current=70.0)],
            "gpu": [MagicMock(current=55.0)],
        }
        with patch(PATCH_TARGET, create=True, return_value=mock_temps):
            monitor = ThermalMonitor()
            monitor.enabled = True
            temp = monitor._get_max_temp()
        assert temp == 70.0
