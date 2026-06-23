from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

from .base import SensorBase
from .registry import SensorRegistry


def _single_cell_voltage_to_percent(voltage_v: float) -> int:
    """Piecewise linear SoC estimate for a single Li-ion/LiPo cell."""
    if voltage_v >= 4.20:
        return 100
    if voltage_v >= 4.00:
        return int(85 + (voltage_v - 4.00) / 0.20 * 15)
    if voltage_v >= 3.80:
        return int(60 + (voltage_v - 3.80) / 0.20 * 25)
    if voltage_v >= 3.70:
        return int(40 + (voltage_v - 3.70) / 0.10 * 20)
    if voltage_v >= 3.50:
        return int(15 + (voltage_v - 3.50) / 0.20 * 25)
    if voltage_v >= 3.00:
        return int((voltage_v - 3.00) / 0.50 * 15)
    return 0


@SensorRegistry.register("pymc_modem")
class PymcModemSensor(SensorBase):
    """Read diagnostics exposed by a pyMC modem HTTP API."""

    sensor_type = "pymc_modem"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, log=None):
        super().__init__(name=name, config=config, log=log)
        self.poll_interval_seconds = float(self.settings.get("poll_interval_seconds", 60.0))
        self.timeout_seconds = float(self.settings.get("timeout_seconds", 2.0))
        self.endpoint = str(self.settings.get("endpoint", "/api/stats") or "/api/stats")
        self.url = self._build_url()
        self.username = str(self.settings.get("username", "admin") or "admin")
        self.password = self.settings.get("password")

    def _build_url(self) -> str:
        base_url = self.settings.get("base_url")
        if base_url:
            base = str(base_url).rstrip("/") + "/"
            return self._validate_url(urljoin(base, self.endpoint.lstrip("/")))

        host = str(self.settings.get("host", "") or "").strip()
        if not host:
            raise ValueError("pymc_modem requires settings.host or settings.base_url")
        scheme = str(self.settings.get("scheme", "http") or "http").lower()
        if scheme not in {"http", "https"}:
            raise ValueError("pymc_modem scheme must be http or https")
        port = self.settings.get("port")
        netloc = host
        if port not in (None, ""):
            netloc = f"{host}:{int(port)}"
        return self._validate_url(
            f"{scheme}://{netloc}{self.endpoint if self.endpoint.startswith('/') else '/' + self.endpoint}"
        )

    @staticmethod
    def _validate_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("pymc_modem URL scheme must be http or https")
        if not parsed.netloc:
            raise ValueError("pymc_modem URL must include a host")
        return url

    def _read(self) -> Dict[str, Any]:
        request = urllib.request.Request(self.url, headers={"Accept": "application/json"})
        if self.password not in (None, ""):
            raw = f"{self.username}:{self.password}".encode("utf-8")
            request.add_header("Authorization", "Basic " + base64.b64encode(raw).decode("ascii"))

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310
                status = int(getattr(response, "status", 200) or 200)
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"pyMC modem HTTP {exc.code} reading {self.url}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"pyMC modem request failed: {exc.reason}") from exc

        if status < 200 or status >= 300:
            raise RuntimeError(f"pyMC modem HTTP {status} reading {self.url}")

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("pyMC modem response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("pyMC modem response was not a JSON object")

        return self._normalize_payload(payload)

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_gps = payload.get("gps")
        gps: Dict[str, Any] = raw_gps if isinstance(raw_gps, dict) else {}
        position = self._first_dict(
            gps.get("position"),
            gps.get("gps_position"),
            gps.get("location"),
            payload.get("gps_position"),
            payload.get("position"),
            payload,
        )
        fix = self._first_dict(gps.get("fix"), payload.get("fix"))
        satellites = self._first_dict(gps.get("satellites"), payload.get("satellites"))
        time_data = self._first_dict(
            gps.get("time"), gps.get("time_data"), payload.get("time_data")
        )
        motion = self._first_dict(gps.get("motion"), payload.get("motion"))

        out: Dict[str, Any] = {
            "source": "pymc_modem",
            "url": self.url,
            "gps_enabled": self._bool_or_none(gps.get("enabled")),
            "gps_seen": self._bool_or_none(gps.get("seen")),
            "latitude": self._float(position.get("latitude")),
            "longitude": self._float(position.get("longitude")),
            "altitude_m": self._float(position.get("altitude_m")),
            "fix_valid": self._bool_or_none(fix.get("valid")),
            "fix_quality": self._int(fix.get("quality")),
            "satellites_used": self._int(
                satellites.get("used_count", satellites.get("satellites_used"))
            ),
            "satellites_in_view": self._int(
                satellites.get("in_view_count", satellites.get("satellites_in_view"))
            ),
            "datetime_utc": time_data.get("datetime_utc") or payload.get("datetime_utc"),
            "speed_kmh": self._float(motion.get("speed_kmh", payload.get("speed_kmh"))),
            "course_degrees": self._float(
                motion.get("course_degrees", payload.get("course_degrees"))
            ),
        }

        for key in (
            "battery_voltage_mv",
            "battery_voltage_v",
            "battery_percent",
            "battery_percentage",
            "solar_charge_rate_percent_per_hour",
        ):
            if key in payload:
                out[key] = payload[key]

        if "battery_percent" not in out:
            battery_voltage_v = self._battery_voltage_v(payload)
            if battery_voltage_v is not None:
                out["battery_percent"] = _single_cell_voltage_to_percent(battery_voltage_v)

        return {key: value for key, value in out.items() if value is not None}

    def _battery_voltage_v(self, payload: Dict[str, Any]) -> Optional[float]:
        voltage_v = self._float(payload.get("battery_voltage_v"))
        if voltage_v is not None:
            return voltage_v
        voltage_mv = self._float(payload.get("battery_voltage_mv"))
        if voltage_mv is not None:
            return voltage_mv / 1000.0
        return None

    @staticmethod
    def _first_dict(*values: Any) -> Dict[str, Any]:
        for value in values:
            if isinstance(value, dict):
                return value
        return {}

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result

    @staticmethod
    def _int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _bool_or_none(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)
