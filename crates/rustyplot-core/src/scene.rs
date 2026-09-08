//! Scene description: what to draw, independent of how it is drawn.

use crate::view::{View2d, Viewport};

/// A set of points with per-point size (in pixels) and RGBA colour.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScatterSeries {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub size: Vec<f32>,
    pub color: Vec<[f32; 4]>,
}

/// Reasons a series cannot be drawn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeriesError {
    LengthMismatch {
        field: &'static str,
        expected: usize,
        found: usize,
    },
}

impl core::fmt::Display for SeriesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch {
                field,
                expected,
                found,
            } => write!(
                f,
                "`{field}` has {found} elements but `x` has {expected}; all series arrays must be the same length"
            ),
        }
    }
}

impl core::error::Error for SeriesError {}

impl ScatterSeries {
    /// Build a series with a uniform size and colour.
    pub fn new(x: Vec<f32>, y: Vec<f32>, size: f32, color: [f32; 4]) -> Self {
        let n = x.len();
        Self {
            x,
            y,
            size: vec![size; n],
            color: vec![color; n],
        }
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn validate(&self) -> Result<(), SeriesError> {
        let n = self.x.len();
        for (field, found) in [
            ("y", self.y.len()),
            ("size", self.size.len()),
            ("color", self.color.len()),
        ] {
            if found != n {
                return Err(SeriesError::LengthMismatch {
                    field,
                    expected: n,
                    found,
                });
            }
        }
        Ok(())
    }

    /// Index of the point closest to a pixel position, within `max_px`.
    ///
    /// Brute force in pixel space; this is the interaction hot path and stays in Rust
    /// so that hit-testing never crosses the Python boundary.
    pub fn nearest(
        &self,
        px: f32,
        py: f32,
        view: &View2d,
        vp: Viewport,
        max_px: f32,
    ) -> Option<(usize, f32)> {
        let sx = vp.width / view.width();
        let sy = vp.height / view.height();
        let max_sq = max_px * max_px;
        let mut best: Option<(usize, f32)> = None;
        for i in 0..self.len() {
            let x = self.x[i];
            let y = self.y[i];
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            let dx = (x - view.x_min) * sx - px;
            let dy = (view.y_max - y) * sy - py;
            let d2 = dx * dx + dy * dy;
            if d2 <= max_sq && best.is_none_or(|(_, b)| d2 < b) {
                best = Some((i, d2));
            }
        }
        best.map(|(i, d2)| (i, d2.sqrt()))
    }
}

/// Everything needed to produce one frame.
#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub background: [f32; 4],
    pub view: View2d,
    pub scatter: Vec<ScatterSeries>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            background: [1.0, 1.0, 1.0, 1.0],
            view: View2d::default(),
            scatter: Vec::new(),
        }
    }

    pub fn point_count(&self) -> usize {
        self.scatter.iter().map(ScatterSeries::len).sum()
    }

    /// Fit the view to all series currently in the scene.
    pub fn autoscale(&mut self, margin: f32) {
        let mut x: Vec<f32> = Vec::new();
        let mut y: Vec<f32> = Vec::new();
        for s in &self.scatter {
            x.extend_from_slice(&s.x);
            y.extend_from_slice(&s.y);
        }
        if !x.is_empty() {
            self.view = View2d::from_points(&x, &y, margin);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_catches_length_mismatch() {
        let mut s = ScatterSeries::new(vec![0.0, 1.0], vec![0.0, 1.0], 4.0, [0.0; 4]);
        s.size.pop();
        assert!(matches!(
            s.validate(),
            Err(SeriesError::LengthMismatch { field: "size", .. })
        ));
    }

    #[test]
    fn nearest_finds_the_closest_point_within_tolerance() {
        let s = ScatterSeries::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0], 4.0, [0.0; 4]);
        let view = View2d::new(0.0, 2.0, 0.0, 2.0);
        let vp = Viewport::new(100.0, 100.0);
        // Data (1,1) sits at the centre of a 100x100 viewport.
        let hit = s.nearest(50.0, 50.0, &view, vp, 10.0);
        assert_eq!(hit.map(|(i, _)| i), Some(1));
    }

    #[test]
    fn nearest_returns_none_beyond_tolerance() {
        let s = ScatterSeries::new(vec![0.0], vec![0.0], 4.0, [0.0; 4]);
        let view = View2d::new(0.0, 2.0, 0.0, 2.0);
        let vp = Viewport::new(100.0, 100.0);
        assert!(s.nearest(50.0, 50.0, &view, vp, 5.0).is_none());
    }

    #[test]
    fn autoscale_covers_all_series() {
        let mut scene = Scene::new();
        scene
            .scatter
            .push(ScatterSeries::new(vec![0.0], vec![0.0], 1.0, [0.0; 4]));
        scene
            .scatter
            .push(ScatterSeries::new(vec![10.0], vec![4.0], 1.0, [0.0; 4]));
        scene.autoscale(0.0);
        assert_eq!((scene.view.x_min, scene.view.x_max), (0.0, 10.0));
        assert_eq!((scene.view.y_min, scene.view.y_max), (0.0, 4.0));
        assert_eq!(scene.point_count(), 2);
    }
}
