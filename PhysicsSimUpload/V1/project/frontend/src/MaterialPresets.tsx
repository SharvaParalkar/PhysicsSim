import React, { useMemo } from "react";

const MATERIAL_PRESETS = [
  { label: "Steel", E: 200_000_000 },
  { label: "Hard plastic", E: 50_000_000 },
  { label: "Hard rubber", E: 1_000_000 },
  { label: "Silicone", E: 10_000 },
  { label: "Gel", E: 1_000 }
];

type MaterialPresetSliderProps = {
  value: number;
  onChangeE: (value: number) => void;
};

export const MaterialPresetSlider: React.FC<MaterialPresetSliderProps> = ({
  value,
  onChangeE
}) => {
  const sliderValue = useMemo(() => Math.log10(value || 1), [value]);

  const nearestPreset = useMemo(() => {
    const logE = sliderValue;
    let best = MATERIAL_PRESETS[0];
    let bestDist = Math.abs(Math.log10(best.E) - logE);
    for (const p of MATERIAL_PRESETS) {
      const d = Math.abs(Math.log10(p.E) - logE);
      if (d < bestDist) {
        best = p;
        bestDist = d;
      }
    }
    return best;
  }, [sliderValue]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = parseFloat(e.target.value);
    const E = 10 ** v;
    onChangeE(E);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <label>
        Material stiffness (log₁₀E, Pa)
        <input
          type="range"
          min={3}
          max={8}
          step={0.01}
          value={sliderValue}
          onChange={handleChange}
          style={{ width: "100%" }}
        />
      </label>
      <div>
        <strong>Preset:</strong> {nearestPreset.label} &mdash; E ≈{" "}
        {nearestPreset.E.toLocaleString()} Pa
      </div>
      <div>
        <strong>Current E:</strong> {Math.round(value).toLocaleString()} Pa
      </div>
    </div>
  );
};

