//! Scene description: what to draw, independent of how it is drawn.

use crate::axes::Axes2d;
use crate::view::{Rect, View2d, Viewport};

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

/// Everything needed to produce one frame: a background and a grid of axes.
///
/// A 1x1 grid (the default) is still a grid; it just has one cell. This is
/// deliberate (see `AGENTS.md`): retrofitting multiple viewports onto a
/// single-view core later would be far more invasive than starting here.
#[derive(Clone, Debug)]
pub struct Scene {
    pub background: [f32; 4],
    pub nrows: usize,
    pub ncols: usize,
    pub axes: Vec<Axes2d>,
}

impl Default for Scene {
    fn default() -> Self {
        Self::grid(1, 1)
    }
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    /// A figure with `nrows * ncols` axes, laid out row-major.
    pub fn grid(nrows: usize, ncols: usize) -> Self {
        let nrows = nrows.max(1);
        let ncols = ncols.max(1);
        Self {
            background: [1.0, 1.0, 1.0, 1.0],
            nrows,
            ncols,
            axes: (0..nrows * ncols).map(|_| Axes2d::default()).collect(),
        }
    }

    pub fn axes(&self, row: usize, col: usize) -> &Axes2d {
        &self.axes[row * self.ncols + col]
    }

    pub fn axes_mut(&mut self, row: usize, col: usize) -> &mut Axes2d {
        &mut self.axes[row * self.ncols + col]
    }

    /// The single axes of a 1x1 figure. Convenience for the common case while
    /// larger grids are not yet exposed to Python.
    pub fn primary(&self) -> &Axes2d {
        &self.axes[0]
    }

    pub fn primary_mut(&mut self) -> &mut Axes2d {
        &mut self.axes[0]
    }

    pub fn point_count(&self) -> usize {
        self.axes.iter().map(Axes2d::point_count).sum()
    }

    /// Fit each axes' view to its own series.
    pub fn autoscale(&mut self, margin: f32) {
        for axes in &mut self.axes {
            axes.autoscale(margin);
        }
    }

    /// Split `figure` into an `nrows x ncols` grid of equal cells and lay out
    /// each axes (reserving margins for its own ticks and labels) within its cell.
    pub fn layout(&mut self, figure: Viewport) {
        let cell_width = figure.width / self.ncols as f32;
        let cell_height = figure.height / self.nrows as f32;
        for row in 0..self.nrows {
            for col in 0..self.ncols {
                let cell = Rect::new(
                    col as f32 * cell_width,
                    row as f32 * cell_height,
                    cell_width,
                    cell_height,
                );
                self.axes_mut(row, col).layout(cell);
            }
        }
    }

    /// Index of the axes whose plot area contains a pixel position, if any.
    pub fn axes_at(&self, px: f32, py: f32) -> Option<usize> {
        self.axes.iter().position(|a| a.rect.contains(px, py))
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
    fn autoscale_covers_all_series_in_one_axes() {
        let mut scene = Scene::new();
        scene
            .primary_mut()
            .scatter
            .push(ScatterSeries::new(vec![0.0], vec![0.0], 1.0, [0.0; 4]));
        scene
            .primary_mut()
            .scatter
            .push(ScatterSeries::new(vec![10.0], vec![4.0], 1.0, [0.0; 4]));
        scene.autoscale(0.0);
        let view = scene.primary().view;
        assert_eq!((view.x_min, view.x_max), (0.0, 10.0));
        assert_eq!((view.y_min, view.y_max), (0.0, 4.0));
        assert_eq!(scene.point_count(), 2);
    }

    #[test]
    fn grid_indexes_axes_row_major() {
        let mut scene = Scene::grid(2, 3);
        scene.axes_mut(1, 2).title = "bottom-right".into();
        assert_eq!(scene.axes(1, 2).title, "bottom-right");
        assert_eq!(scene.axes.len(), 6);
    }

    #[test]
    fn layout_splits_figure_into_equal_cells() {
        let mut scene = Scene::grid(1, 2);
        scene.layout(Viewport::new(800.0, 400.0));
        // Each axes' plot area (inside margins) stays within its half of the figure.
        assert!(scene.axes(0, 0).rect.x + scene.axes(0, 0).rect.width <= 400.0);
        assert!(scene.axes(0, 1).rect.x >= 400.0);
    }

    #[test]
    fn axes_at_finds_the_containing_axes() {
        let mut scene = Scene::grid(1, 2);
        scene.layout(Viewport::new(800.0, 400.0));
        let left_center = (scene.axes(0, 0).rect.x + 1.0, scene.axes(0, 0).rect.y + 1.0);
        let right_center = (scene.axes(0, 1).rect.x + 1.0, scene.axes(0, 1).rect.y + 1.0);
        assert_eq!(scene.axes_at(left_center.0, left_center.1), Some(0));
        assert_eq!(scene.axes_at(right_center.0, right_center.1), Some(1));
    }
}
