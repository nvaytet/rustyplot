//! Tick placement and formatting, shared by layout (to size margins) and
//! rendering (to place spines, tick marks and labels) so the two always agree.

/// Pick "nice" tick positions covering `[min, max]`, aiming for `target` ticks.
///
/// Step sizes are `{1, 2, 5} * 10^n`, matplotlib/R's classic scheme: it keeps
/// labels short and round rather than showing the requested count exactly.
pub fn nice_ticks(min: f32, max: f32, target: usize) -> Vec<f32> {
    if target == 0 || !(max > min) || !min.is_finite() || !max.is_finite() {
        return Vec::new();
    }

    let step = nice_step((max - min) / target as f32);
    if !(step > 0.0) {
        return Vec::new();
    }

    let start = (min / step).floor() * step;
    // Generous bound: a handful more than requested, so float drift near the
    // upper edge can't silently truncate the last tick.
    let max_count = target * 4 + 10;
    let epsilon = step * 1e-4;

    let mut ticks = Vec::new();
    let mut v = start;
    let mut i = 0;
    while v <= max + epsilon && i < max_count {
        if v >= min - epsilon {
            // Snap away the float noise that `start + n*step` accumulates.
            ticks.push((v / step).round() * step);
        }
        i += 1;
        v = start + i as f32 * step;
    }
    ticks
}

/// Round `raw_step` up to the nearest `{1, 2, 5} * 10^n`.
fn nice_step(raw_step: f32) -> f32 {
    if !(raw_step > 0.0) {
        return 0.0;
    }
    let mag = 10f32.powf(raw_step.log10().floor());
    let residual = raw_step / mag;
    let nice_residual = if residual <= 1.0 {
        1.0
    } else if residual <= 2.0 {
        2.0
    } else if residual <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice_residual * mag
}

/// Format tick values consistently: enough decimal places to represent the
/// tick step itself, no more.
pub fn format_ticks(values: &[f32], step: f32) -> Vec<String> {
    let decimals = decimals_for_step(step);
    values
        .iter()
        .map(|v| {
            let s = format!("{:.*}", decimals, v);
            // Rounding can produce "-0" or "-0.00" for values that were only
            // negative due to float noise.
            if s.trim_start_matches('-').chars().all(|c| c == '0' || c == '.') {
                s.trim_start_matches('-').to_string()
            } else {
                s
            }
        })
        .collect()
}

/// Smallest number of decimal digits (up to a cap) that represents `step`
/// exactly, so a 0.25 step gets "0.25" rather than the "0.3" that one decimal
/// place (chosen from `-log10(step)` alone) would round it to.
fn decimals_for_step(step: f32) -> usize {
    if !(step > 0.0) {
        return 0;
    }
    for d in 0..=6 {
        let scale = 10f32.powi(d);
        if ((step * scale).round() - step * scale).abs() < 1e-3 {
            return d as usize;
        }
    }
    6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_ticks_covers_a_simple_range() {
        let ticks = nice_ticks(0.0, 10.0, 5);
        assert_eq!(ticks, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn nice_ticks_handles_negative_ranges() {
        let ticks = nice_ticks(-5.0, 5.0, 4);
        assert_eq!(ticks, vec![-5.0, 0.0, 5.0]);
    }

    #[test]
    fn nice_ticks_handles_small_fractional_ranges() {
        let ticks = nice_ticks(0.0, 0.09, 4);
        assert!(!ticks.is_empty());
        for w in ticks.windows(2) {
            assert!((w[1] - w[0] - (ticks[1] - ticks[0])).abs() < 1e-6);
        }
    }

    #[test]
    fn nice_ticks_on_degenerate_range_is_empty() {
        assert!(nice_ticks(5.0, 5.0, 5).is_empty());
        assert!(nice_ticks(1.0, 2.0, 0).is_empty());
        assert!(nice_ticks(f32::NAN, 2.0, 5).is_empty());
    }

    #[test]
    fn nice_ticks_stays_within_or_near_bounds() {
        let ticks = nice_ticks(3.0, 27.0, 5);
        assert!(ticks.iter().all(|&t| t >= 0.0 && t <= 30.0));
    }

    #[test]
    fn format_ticks_matches_step_precision() {
        let labels = format_ticks(&[0.0, 2.0, 4.0], 2.0);
        assert_eq!(labels, vec!["0", "2", "4"]);

        let labels = format_ticks(&[0.0, 0.25, 0.5], 0.25);
        assert_eq!(labels, vec!["0.00", "0.25", "0.50"]);
    }

    #[test]
    fn format_ticks_avoids_negative_zero() {
        let labels = format_ticks(&[-0.0001], 1.0);
        assert_eq!(labels, vec!["0"]);
    }
}
