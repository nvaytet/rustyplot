//! Interaction state machine, shared by every frontend so that pan/zoom behave
//! identically on the desktop and in the notebook.

use crate::scene::Scene;

/// Result of hit-testing a pixel position against the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PickHit {
    pub axes: usize,
    pub series: usize,
    pub index: usize,
    pub x: f32,
    pub y: f32,
    pub distance_px: f32,
}

/// Tracks drag state and applies pan/zoom to a scene's axes.
///
/// Pixel positions passed in are figure-absolute (origin top-left of the
/// whole canvas); the axes under the cursor is found via [`Scene::axes_at`].
#[derive(Clone, Debug)]
pub struct Interaction {
    pub pick_radius_px: f32,
    /// Wheel sensitivity: view extent is multiplied by `exp(delta * zoom_rate)`.
    pub zoom_rate: f32,
    /// Pointer movement below this distance still counts as a click, not a drag.
    pub click_slop_px: f32,
    drag: Option<DragState>,
}

#[derive(Clone, Copy, Debug)]
struct DragState {
    axes: usize,
    last: (f32, f32),
    travelled: f32,
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            pick_radius_px: 10.0,
            zoom_rate: 0.0015,
            click_slop_px: 4.0,
            drag: None,
        }
    }
}

impl Interaction {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Begin a drag, locking onto whichever axes is under the cursor. Does
    /// nothing if the position is outside every axes' plot area.
    pub fn pointer_down(&mut self, px: f32, py: f32, scene: &Scene) {
        if let Some(axes) = scene.axes_at(px, py) {
            self.drag = Some(DragState {
                axes,
                last: (px, py),
                travelled: 0.0,
            });
        }
    }

    /// Returns `true` if the view changed and a redraw is needed.
    pub fn pointer_move(&mut self, px: f32, py: f32, scene: &mut Scene) -> bool {
        let Some(drag) = self.drag.as_mut() else {
            return false;
        };
        let dx = px - drag.last.0;
        let dy = py - drag.last.1;
        drag.last = (px, py);
        drag.travelled += (dx * dx + dy * dy).sqrt();
        if dx == 0.0 && dy == 0.0 {
            return false;
        }
        let axes = &mut scene.axes[drag.axes];
        axes.view.pan_by_pixels(dx, dy, axes.rect.viewport());
        true
    }

    /// Ends a drag, reporting whether it should be treated as a click.
    pub fn pointer_up(&mut self) -> bool {
        self.drag
            .take()
            .is_some_and(|d| d.travelled <= self.click_slop_px)
    }

    /// Wheel zoom anchored at the cursor. `delta` follows the browser convention
    /// (positive scrolls down / zooms out). Does nothing outside every axes.
    pub fn wheel(&mut self, px: f32, py: f32, delta: f32, scene: &mut Scene) {
        let Some(idx) = scene.axes_at(px, py) else {
            return;
        };
        let axes = &mut scene.axes[idx];
        let factor = (delta * self.zoom_rate).exp();
        let (lx, ly) = (px - axes.rect.x, py - axes.rect.y);
        axes.view.zoom_at_pixel(lx, ly, factor, axes.rect.viewport());
    }

    pub fn pick(&self, px: f32, py: f32, scene: &Scene) -> Option<PickHit> {
        let axes_idx = scene.axes_at(px, py)?;
        let axes = &scene.axes[axes_idx];
        let (lx, ly) = (px - axes.rect.x, py - axes.rect.y);
        let mut best: Option<PickHit> = None;
        for (s, series) in axes.scatter.iter().enumerate() {
            let Some((index, distance_px)) =
                series.nearest(lx, ly, &axes.view, axes.rect.viewport(), self.pick_radius_px)
            else {
                continue;
            };
            if best.is_none_or(|b| distance_px < b.distance_px) {
                best = Some(PickHit {
                    axes: axes_idx,
                    series: s,
                    index,
                    x: series.x[index],
                    y: series.y[index],
                    distance_px,
                });
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::ScatterSeries;
    use crate::view::{Rect, View2d, Viewport};

    fn scene() -> Scene {
        let mut s = Scene::new();
        {
            let axes = s.primary_mut();
            axes.view = View2d::new(0.0, 2.0, 0.0, 2.0);
            axes.rect = Rect::new(0.0, 0.0, 100.0, 100.0);
            axes.scatter.push(ScatterSeries::new(
                vec![0.0, 1.0, 2.0],
                vec![0.0, 1.0, 2.0],
                4.0,
                [0.0; 4],
            ));
        }
        s
    }

    fn interaction() -> Interaction {
        Interaction::new()
    }

    #[test]
    fn drag_pans_the_view() {
        let mut i = interaction();
        let mut s = scene();
        let before = s.primary().view;
        i.pointer_down(50.0, 50.0, &s);
        assert!(i.pointer_move(60.0, 50.0, &mut s));
        assert!(s.primary().view.x_min < before.x_min);
    }

    #[test]
    fn move_without_press_does_nothing() {
        let mut i = interaction();
        let mut s = scene();
        let before = s.primary().view;
        assert!(!i.pointer_move(60.0, 50.0, &mut s));
        assert_eq!(s.primary().view, before);
    }

    #[test]
    fn pointer_down_outside_any_axes_does_not_start_a_drag() {
        let mut i = interaction();
        let s = scene();
        i.pointer_down(500.0, 500.0, &s);
        assert!(!i.is_dragging());
    }

    #[test]
    fn small_movement_still_counts_as_a_click() {
        let mut i = interaction();
        let mut s = scene();
        i.pointer_down(50.0, 50.0, &s);
        i.pointer_move(51.0, 50.0, &mut s);
        assert!(i.pointer_up());
    }

    #[test]
    fn dragging_suppresses_the_click() {
        let mut i = interaction();
        let mut s = scene();
        i.pointer_down(10.0, 10.0, &s);
        i.pointer_move(80.0, 10.0, &mut s);
        assert!(!i.pointer_up());
    }

    #[test]
    fn wheel_down_zooms_out() {
        let mut i = interaction();
        let mut s = scene();
        let before = s.primary().view.width();
        i.wheel(50.0, 50.0, 100.0, &mut s);
        assert!(s.primary().view.width() > before);
    }

    #[test]
    fn wheel_up_zooms_in() {
        let mut i = interaction();
        let mut s = scene();
        let before = s.primary().view.width();
        i.wheel(50.0, 50.0, -100.0, &mut s);
        assert!(s.primary().view.width() < before);
    }

    #[test]
    fn pick_reports_the_data_coordinates() {
        let i = interaction();
        let s = scene();
        let hit = i.pick(50.0, 50.0, &s).expect("centre point should be hit");
        assert_eq!(hit.axes, 0);
        assert_eq!(hit.index, 1);
        assert_eq!((hit.x, hit.y), (1.0, 1.0));
    }

    #[test]
    fn pick_misses_empty_space() {
        let i = interaction();
        let s = scene();
        assert!(i.pick(25.0, 75.0, &s).is_none());
    }

    #[test]
    fn pick_outside_any_axes_is_none() {
        let i = interaction();
        let s = scene();
        assert!(i.pick(500.0, 500.0, &s).is_none());
    }
}
