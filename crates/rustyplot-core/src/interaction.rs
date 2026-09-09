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

/// Which gesture a drag/scroll currently performs. Toolbar buttons in every
/// frontend (native and notebook) just call [`Interaction::set_mode`]; the
/// gating below is the single place that decides what pan/box-zoom/scroll
/// actually do, so the two frontends can never drift apart (see AGENTS.md's
/// "interaction logic lives in `rustyplot-core::interaction`").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractionMode {
    /// Drag pans the axes under the cursor.
    Pan,
    /// The mouse wheel zooms at the cursor.
    ZoomScroll,
    /// Drag draws a rectangle; releasing zooms the view to it.
    BoxZoom,
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
    /// The active tool, or `None` if no toolbar button is toggled on (in
    /// which case dragging and the wheel do nothing; a plain click still
    /// reaches [`Interaction::pick`] via `pointer_up`'s click/drag split).
    mode: Option<InteractionMode>,
    drag: Option<DragState>,
}

#[derive(Clone, Copy, Debug)]
struct DragState {
    axes: usize,
    /// Pixel position at `pointer_down`, kept (in addition to `last`) so
    /// box-zoom can report the drag's full rectangle, not just its latest
    /// step -- unlike pan, which only ever needs the delta since last move.
    start: (f32, f32),
    last: (f32, f32),
    travelled: f32,
}

/// Normalizes two corner points into `(x, y, width, height)` with a
/// top-left origin and non-negative extents, regardless of drag direction.
fn rect_from(a: (f32, f32), b: (f32, f32)) -> (f32, f32, f32, f32) {
    (a.0.min(b.0), a.1.min(b.1), (b.0 - a.0).abs(), (b.1 - a.1).abs())
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            pick_radius_px: 10.0,
            zoom_rate: 0.0015,
            click_slop_px: 4.0,
            mode: None,
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

    pub fn mode(&self) -> Option<InteractionMode> {
        self.mode
    }

    /// Switches the active tool (or clears it, with `None`). Cancels any
    /// drag in progress, so switching tools mid-drag can't leave a pan or a
    /// box-zoom half-applied.
    pub fn set_mode(&mut self, mode: Option<InteractionMode>) {
        self.mode = mode;
        self.drag = None;
    }

    /// Begin a drag, locking onto whichever axes is under the cursor. Does
    /// nothing if the position is outside every axes' plot area.
    pub fn pointer_down(&mut self, px: f32, py: f32, scene: &Scene) {
        if let Some(axes) = scene.axes_at(px, py) {
            self.drag = Some(DragState {
                axes,
                start: (px, py),
                last: (px, py),
                travelled: 0.0,
            });
        }
    }

    /// Returns `true` if the view (or, in box-zoom mode, the in-progress
    /// selection rectangle) changed and a redraw is needed.
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
        match self.mode {
            Some(InteractionMode::Pan) => {
                let axes = &mut scene.axes[drag.axes];
                axes.view.pan_by_pixels(dx, dy, axes.rect.viewport());
                true
            }
            // The rectangle itself is only ever drawn from `drag_rect`, not
            // stored anywhere else, so there is nothing to update here
            // besides `drag.last` above; the redraw just re-reads it.
            Some(InteractionMode::BoxZoom) => true,
            Some(InteractionMode::ZoomScroll) | None => false,
        }
    }

    /// Ends a drag at `(px, py)` -- the pointer's actual release position,
    /// which may be beyond the last `pointer_move` the browser delivered --
    /// applying a box-zoom if that was the active tool and the gesture was
    /// a genuine drag rather than a click. Reports whether it should be
    /// treated as a click.
    pub fn pointer_up(&mut self, px: f32, py: f32, scene: &mut Scene) -> bool {
        let Some(mut drag) = self.drag.take() else {
            return false;
        };
        let dx = px - drag.last.0;
        let dy = py - drag.last.1;
        drag.travelled += (dx * dx + dy * dy).sqrt();
        drag.last = (px, py);
        let was_click = drag.travelled <= self.click_slop_px;
        if self.mode == Some(InteractionMode::BoxZoom) && !was_click {
            let (rx, ry, rw, rh) = rect_from(drag.start, drag.last);
            // Require a rectangle with some visible extent in *both*
            // dimensions (not just non-zero): a drag that is almost, but
            // not exactly, axis-aligned would otherwise zoom to a sliver a
            // fraction of a pixel tall or wide, leaving the axes unusable.
            if rw > self.click_slop_px && rh > self.click_slop_px {
                let axes = &mut scene.axes[drag.axes];
                let vp = axes.rect.viewport();
                let (x0, y0) = axes.view.screen_to_data(rx - axes.rect.x, ry - axes.rect.y, vp);
                let (x1, y1) =
                    axes
                        .view
                        .screen_to_data(rx + rw - axes.rect.x, ry + rh - axes.rect.y, vp);
                axes.view = crate::view::View2d::new(x0.min(x1), x0.max(x1), y0.min(y1), y0.max(y1));
            }
        }
        was_click
    }

    /// Aborts a drag in progress without applying anything -- for a
    /// `pointercancel` (e.g. the OS interrupts the gesture), which must not
    /// commit a box-zoom or count as a click the way `pointer_up` would.
    pub fn cancel(&mut self) {
        self.drag = None;
    }

    /// Wheel zoom anchored at the cursor. `delta` follows the browser convention
    /// (positive scrolls down / zooms out). Does nothing unless `ZoomScroll` is
    /// the active mode, or the position is outside every axes.
    ///
    /// Returns `true` if the view changed and a redraw is needed.
    pub fn wheel(&mut self, px: f32, py: f32, delta: f32, scene: &mut Scene) -> bool {
        if self.mode != Some(InteractionMode::ZoomScroll) {
            return false;
        }
        let Some(idx) = scene.axes_at(px, py) else {
            return false;
        };
        let axes = &mut scene.axes[idx];
        let factor = (delta * self.zoom_rate).exp();
        let (lx, ly) = (px - axes.rect.x, py - axes.rect.y);
        axes.view.zoom_at_pixel(lx, ly, factor, axes.rect.viewport());
        true
    }

    /// The in-progress box-zoom selection rectangle, as
    /// `(axes, x, y, width, height)` in figure-absolute screen pixels, for a
    /// frontend to draw as a marquee overlay. `None` unless box-zoom is the
    /// active mode and a drag is under way.
    pub fn drag_rect(&self) -> Option<(usize, f32, f32, f32, f32)> {
        if self.mode != Some(InteractionMode::BoxZoom) {
            return None;
        }
        let drag = self.drag.as_ref()?;
        let (x, y, w, h) = rect_from(drag.start, drag.last);
        Some((drag.axes, x, y, w, h))
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
        // Line markers are pickable too (see `LineSeries::nearest`); indexed
        // after every scatter series so `series` stays unique within one axes.
        for (s, series) in axes.lines.iter().enumerate() {
            let Some((index, distance_px)) =
                series.nearest(lx, ly, &axes.view, axes.rect.viewport(), self.pick_radius_px)
            else {
                continue;
            };
            if best.is_none_or(|b| distance_px < b.distance_px) {
                best = Some(PickHit {
                    axes: axes_idx,
                    series: axes.scatter.len() + s,
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
    use crate::view::{Rect, View2d};

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
    fn drag_pans_the_view_in_pan_mode() {
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::Pan));
        let mut s = scene();
        let before = s.primary().view;
        i.pointer_down(50.0, 50.0, &s);
        assert!(i.pointer_move(60.0, 50.0, &mut s));
        assert!(s.primary().view.x_min < before.x_min);
    }

    #[test]
    fn drag_does_nothing_without_a_mode() {
        let mut i = interaction();
        let mut s = scene();
        let before = s.primary().view;
        i.pointer_down(50.0, 50.0, &s);
        assert!(!i.pointer_move(60.0, 50.0, &mut s));
        assert_eq!(s.primary().view, before);
    }

    #[test]
    fn move_without_press_does_nothing() {
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::Pan));
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
        assert!(i.pointer_up(51.0, 50.0, &mut s));
    }

    #[test]
    fn dragging_suppresses_the_click() {
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::Pan));
        let mut s = scene();
        i.pointer_down(10.0, 10.0, &s);
        i.pointer_move(80.0, 10.0, &mut s);
        assert!(!i.pointer_up(80.0, 10.0, &mut s));
    }

    #[test]
    fn set_mode_cancels_an_in_progress_drag() {
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::Pan));
        let s = scene();
        i.pointer_down(50.0, 50.0, &s);
        assert!(i.is_dragging());
        i.set_mode(Some(InteractionMode::BoxZoom));
        assert!(!i.is_dragging());
    }

    #[test]
    fn wheel_does_nothing_unless_zoom_scroll_is_active() {
        let mut i = interaction();
        let mut s = scene();
        let before = s.primary().view.width();
        assert!(!i.wheel(50.0, 50.0, 100.0, &mut s));
        assert_eq!(s.primary().view.width(), before);
    }

    #[test]
    fn wheel_down_zooms_out() {
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::ZoomScroll));
        let mut s = scene();
        let before = s.primary().view.width();
        assert!(i.wheel(50.0, 50.0, 100.0, &mut s));
        assert!(s.primary().view.width() > before);
    }

    #[test]
    fn box_zoom_sets_the_view_to_the_dragged_rectangle_on_release() {
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::BoxZoom));
        let mut s = scene();
        // Drag from the bottom-left quarter of the axes to its centre;
        // screen y increases downward while data y increases upward, so
        // the lower-left pixel corner maps to the lower data y.
        i.pointer_down(25.0, 75.0, &s);
        i.pointer_move(75.0, 25.0, &mut s);
        assert!(!i.pointer_up(75.0, 25.0, &mut s), "a real drag is not a click");
        assert_eq!((s.primary().view.x_min, s.primary().view.x_max), (0.5, 1.5));
        assert_eq!((s.primary().view.y_min, s.primary().view.y_max), (0.5, 1.5));
    }

    #[test]
    fn box_zoom_uses_the_actual_release_position_not_just_the_last_move() {
        // The browser can deliver a `pointerup` beyond the last
        // `pointermove` it dispatched; the applied rectangle must reflect
        // where the pointer was actually released.
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::BoxZoom));
        let mut s = scene();
        i.pointer_down(25.0, 75.0, &s);
        i.pointer_move(60.0, 40.0, &mut s);
        i.pointer_up(75.0, 25.0, &mut s);
        assert_eq!((s.primary().view.x_min, s.primary().view.x_max), (0.5, 1.5));
        assert_eq!((s.primary().view.y_min, s.primary().view.y_max), (0.5, 1.5));
    }

    #[test]
    fn box_zoom_ignores_a_near_degenerate_rectangle() {
        // A drag that is almost, but not exactly, axis-aligned must not
        // zoom to a sliver a fraction of a pixel tall.
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::BoxZoom));
        let mut s = scene();
        i.pointer_down(25.0, 50.0, &s);
        i.pointer_move(75.0, 50.5, &mut s);
        let before = s.primary().view;
        i.pointer_up(75.0, 50.5, &mut s);
        assert_eq!(s.primary().view, before);
    }

    #[test]
    fn box_zoom_ignores_a_click_without_a_drag() {
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::BoxZoom));
        let mut s = scene();
        let before = s.primary().view;
        i.pointer_down(50.0, 50.0, &s);
        assert!(i.pointer_up(50.0, 50.0, &mut s));
        assert_eq!(s.primary().view, before);
    }

    #[test]
    fn cancel_discards_the_drag_without_applying_a_box_zoom() {
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::BoxZoom));
        let mut s = scene();
        let before = s.primary().view;
        i.pointer_down(25.0, 75.0, &s);
        i.pointer_move(75.0, 25.0, &mut s);
        i.cancel();
        assert!(!i.is_dragging());
        assert_eq!(s.primary().view, before);
        // A subsequent `pointer_up` (e.g. a stray event) must be a no-op,
        // not reapply anything from the cancelled drag.
        assert!(!i.pointer_up(75.0, 25.0, &mut s));
        assert_eq!(s.primary().view, before);
    }

    #[test]
    fn box_zoom_drag_rect_is_reported_only_while_dragging_in_box_zoom_mode() {
        let mut i = interaction();
        let s = scene();
        assert!(i.drag_rect().is_none());
        i.set_mode(Some(InteractionMode::BoxZoom));
        i.pointer_down(25.0, 75.0, &s);
        let (axes, x, y, w, h) = i.drag_rect().expect("dragging in box-zoom mode");
        assert_eq!(axes, 0);
        assert_eq!((x, y, w, h), (25.0, 75.0, 0.0, 0.0));
        let mut s2 = scene();
        i.pointer_move(75.0, 25.0, &mut s2);
        let (_, x, y, w, h) = i.drag_rect().expect("still dragging");
        assert_eq!((x, y, w, h), (25.0, 25.0, 50.0, 50.0));
    }

    #[test]
    fn pan_mode_does_not_report_a_drag_rect() {
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::Pan));
        let s = scene();
        i.pointer_down(25.0, 75.0, &s);
        assert!(i.drag_rect().is_none());
    }

    #[test]
    fn wheel_up_zooms_in() {
        let mut i = interaction();
        i.set_mode(Some(InteractionMode::ZoomScroll));
        let mut s = scene();
        let before = s.primary().view.width();
        assert!(i.wheel(50.0, 50.0, -100.0, &mut s));
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

    #[test]
    fn pick_finds_a_line_marker_when_no_scatter_point_is_closer() {
        use crate::scene::{LineSeries, Marker, MarkerStyle};
        let mut s = Scene::new();
        {
            let axes = s.primary_mut();
            axes.view = View2d::new(0.0, 2.0, 0.0, 2.0);
            axes.rect = Rect::new(0.0, 0.0, 100.0, 100.0);
            axes.lines.push(LineSeries {
                x: vec![1.0],
                y: vec![1.0],
                color: [0.0; 4],
                line: None,
                marker: Some(Marker {
                    style: MarkerStyle::Circle,
                    size: 6.0,
                }),
            });
        }
        let i = interaction();
        let hit = i.pick(50.0, 50.0, &s).expect("centre marker should be hit");
        assert_eq!(hit.axes, 0);
        // No scatter series in this axes, so the line series' offset is 0.
        assert_eq!(hit.series, 0);
        assert_eq!((hit.x, hit.y), (1.0, 1.0));
    }
}
