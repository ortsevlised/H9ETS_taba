import json
import time
from pathlib import Path

from codecarbon import OfflineEmissionsTracker

from common import read_codecarbon_energy_kwh

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs"
DURATION_S = 15.0


def main():
    csv_path = OUT_DIR / "codecarbon_idle.csv"
    tracker = OfflineEmissionsTracker(
        country_iso_code="IRL",
        project_name="taba_idle_baseline",
        output_dir=str(OUT_DIR),
        output_file=csv_path.name,
        log_level="error",
        measure_power_secs=1,
    )
    tracker.start()
    start = time.perf_counter()
    time.sleep(DURATION_S)
    elapsed_s = time.perf_counter() - start
    emissions_kg = tracker.stop()
    energy_kwh = read_codecarbon_energy_kwh(csv_path)

    result = {
        "duration_s": elapsed_s,
        "emissions_kg_co2e": emissions_kg,
        "energy_kwh": energy_kwh,
        "energy_kwh_per_1000s": (energy_kwh / elapsed_s) * 1000 if energy_kwh is not None else None,
    }
    with open(OUT_DIR / "idle_baseline.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(result)


if __name__ == "__main__":
    main()
