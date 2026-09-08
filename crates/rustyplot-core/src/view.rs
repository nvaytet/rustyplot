//! View transforms: the mapping between data coordinates, screen pixels and NDC.

/// Size of the drawing surface in physical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub width: f32,
    pub height: f32,
}

impl Viewport {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width: width.max(1.0),
            height: height.max(1.0),
        }
    }
}

/// A pixel rectangle, used for an axes' plot area within the figure.
///
/// `(x, y)` is the top-left corner, in the same pixel space as pointer events
/// (origin top-left, y pointing down).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width: width.max(0.0),
            height: height.max(0.0),
        }
    }

    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px <= self.x + self.width && py >= self.y && py <= self.y + self.height
    }

    /// This rect's size as a [`Viewport`], for use with [`View2d`] transforms
    /// that are expressed relative to the rect's own origin.
    pub fn viewport(&self) -> Viewport {
        Viewport::new(self.width, self.height)
    }

    /// Shrink the rect by different amounts on each side. Never produces a
    /// negative size; excess margin is clamped away evenly.
    pub fn shrink(&self, left: f32, right: f32, top: f32, bottom: f32) -> Self {
        let width = (self.width - left - right).max(1.0);
        let height = (self.height - top - bottom).max(1.0);
        Self {
            x: self.x + left,
            y: self.y + top,
            width,
            height,
        }
    }
}

impl Default for Viewport {
    fn default() -> Self {
        Self::new(800.0, 600.0)
    }
}

/// The visible data rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View2d {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
}

impl Default for View2d {
    fn default() -> Self {
        Self::new(0.0, 1.0, 0.0, 1.0)
    }
}

impl View2d {
    pub fn new(x_min: f32, x_max: f32, y_min: f32, y_max: f32) -> Self {
        let mut v = Self {
            x_min,
            x_max,
            y_min,
            y_max,
        };
        v.fix_degenerate();
        v
    }

    /// Fit the view to the given points, padding by `margin` (fraction of extent).
    pub fn from_points(x: &[f32], y: &[f32], margin: f32) -> Self {
        let (x_min, x_max) = finite_bounds(x);
        let (y_min, y_max) = finite_bounds(y);
        let mut view = Self::new(x_min, x_max, y_min, y_max);
        view.expand(margin);
        view
    }

    pub fn width(&self) -> f32 {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> f32 {
        self.y_max - self.y_min
    }

    pub fn center(&self) -> (f32, f32) {
        (
            0.5 * (self.x_min + self.x_max),
            0.5 * (self.y_min + self.y_max),
        )
    }

    /// Multiplicative part of the data -> NDC transform (`ndc = data * scale + offset`).
    pub fn scale(&self) -> [f32; 2] {
        [2.0 / self.width(), 2.0 / self.height()]
    }

    /// Additive part of the data -> NDC transform.
    pub fn offset(&self) -> [f32; 2] {
        let (cx, cy) = self.center();
        let s = self.scale();
        [-cx * s[0], -cy * s[1]]
    }

    /// Convert a pixel position (origin at the top-left, y pointing down) to data coordinates.
    pub fn screen_to_data(&self, px: f32, py: f32, vp: Viewport) -> (f32, f32) {
        (
            self.x_min + (px / vp.width) * self.width(),
            self.y_max - (py / vp.height) * self.height(),
        )
    }

    /// Convert data coordinates to a pixel position (origin at the top-left).
    pub fn data_to_screen(&self, x: f32, y: f32, vp: Viewport) -> (f32, f32) {
        (
            (x - self.x_min) / self.width() * vp.width,
            (self.y_max - y) / self.height() * vp.height,
        )
    }

    /// Translate the view by a mouse displacement in pixels (content follows the cursor).
    pub fn pan_by_pixels(&mut self, dx: f32, dy: f32, vp: Viewport) {
        let sx = self.width() / vp.width;
        let sy = self.height() / vp.height;
        self.x_min -= dx * sx;
        self.x_max -= dx * sx;
        self.y_min += dy * sy;
        self.y_max += dy * sy;
    }

    /// Zoom about a pixel anchor. `factor < 1` zooms in; the anchored data point stays put.
    pub fn zoom_at_pixel(&mut self, px: f32, py: f32, factor: f32, vp: Viewport) {
        let (ax, ay) = self.screen_to_data(px, py, vp);
        let f = factor.clamp(1e-4, 1e4);
        self.x_min = ax - (ax - self.x_min) * f;
        self.x_max = ax + (self.x_max - ax) * f;
        self.y_min = ay - (ay - self.y_min) * f;
        self.y_max = ay + (self.y_max - ay) * f;
        self.fix_degenerate();
    }

    /// Grow the view by a fraction of its current extent on all sides.
    pub fn expand(&mut self, margin: f32) {
        let mx = self.width() * margin;
        let my = self.height() * margin;
        self.x_min -= mx;
        self.x_max += mx;
        self.y_min -= my;
        self.y_max += my;
        self.fix_degenerate();
    }

    fn fix_degenerate(&mut self) {
        if !(self.x_max > self.x_min) {
            let c = if self.x_min.is_finite() { self.x_min } else { 0.0 };
            self.x_min = c - 0.5;
            self.x_max = c + 0.5;
        }
        if !(self.y_max > self.y_min) {
            let c = if self.y_min.is_finite() { self.y_min } else { 0.0 };
            self.y_min = c - 0.5;
            self.y_max = c + 0.5;
        }
    }
}

fn finite_bounds(values: &[f32]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for v in values.iter().copied().filter(|v| v.is_finite()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if lo > hi { (0.0, 1.0) } else { (lo, hi) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VP: Viewport = Viewport {
        width: 100.0,
        height: 100.0,
    };

    #[test]
    fn ndc_transform_maps_corners() {
        let v = View2d::new(0.0, 10.0, 0.0, 4.0);
        let s = v.scale();
        let o = v.offset();
        assert!((v.x_min * s[0] + o[0] + 1.0).abs() < 1e-6);
        assert!((v.x_max * s[0] + o[0] - 1.0).abs() < 1e-6);
        assert!((v.y_min * s[1] + o[1] + 1.0).abs() < 1e-6);
        assert!((v.y_max * s[1] + o[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn screen_and_data_roundtrip() {
        let v = View2d::new(-3.0, 5.0, 10.0, 20.0);
        let (x, y) = v.screen_to_data(25.0, 75.0, VP);
        let (px, py) = v.data_to_screen(x, y, VP);
        assert!((px - 25.0).abs() < 1e-3);
        assert!((py - 75.0).abs() < 1e-3);
    }

    #[test]
    fn screen_y_is_flipped() {
        let v = View2d::new(0.0, 1.0, 0.0, 1.0);
        let (_, top) = v.screen_to_data(0.0, 0.0, VP);
        let (_, bottom) = v.screen_to_data(0.0, VP.height, VP);
        assert!((top - 1.0).abs() < 1e-6);
        assert!((bottom - 0.0).abs() < 1e-6);
    }

    #[test]
    fn zoom_keeps_anchor_fixed() {
        let mut v = View2d::new(0.0, 10.0, 0.0, 10.0);
        let anchor = v.screen_to_data(30.0, 70.0, VP);
        v.zoom_at_pixel(30.0, 70.0, 0.5, VP);
        let after = v.screen_to_data(30.0, 70.0, VP);
        assert!((anchor.0 - after.0).abs() < 1e-4);
        assert!((anchor.1 - after.1).abs() < 1e-4);
        assert!((v.width() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn pan_moves_data_with_cursor() {
        let mut v = View2d::new(0.0, 100.0, 0.0, 100.0);
        v.pan_by_pixels(10.0, 0.0, VP);
        assert!((v.x_min + 10.0).abs() < 1e-4);
        v.pan_by_pixels(0.0, 10.0, VP);
        assert!((v.y_min - 10.0).abs() < 1e-4);
    }

    #[test]
    fn degenerate_ranges_are_repaired() {
        let v = View2d::new(5.0, 5.0, 1.0, 1.0);
        assert!(v.width() > 0.0);
        assert!(v.height() > 0.0);
    }

    #[test]
    fn rect_contains_checks_bounds() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert!(r.contains(10.0, 20.0));
        assert!(r.contains(110.0, 70.0));
        assert!(!r.contains(9.0, 20.0));
        assert!(!r.contains(10.0, 70.1));
    }

    #[test]
    fn rect_shrink_moves_origin_and_reduces_size() {
        let r = Rect::new(0.0, 0.0, 200.0, 100.0);
        let inner = r.shrink(20.0, 5.0, 10.0, 30.0);
        assert_eq!((inner.x, inner.y), (20.0, 10.0));
        assert_eq!((inner.width, inner.height), (175.0, 60.0));
    }

    #[test]
    fn rect_shrink_never_goes_negative() {
        let r = Rect::new(0.0, 0.0, 10.0, 10.0);
        let inner = r.shrink(20.0, 20.0, 20.0, 20.0);
        assert!(inner.width >= 1.0 && inner.height >= 1.0);
    }

    #[test]
    fn bounds_ignore_non_finite_values() {
        let v = View2d::from_points(&[1.0, f32::NAN, 3.0], &[0.0, f32::INFINITY, 2.0], 0.0);
        assert_eq!((v.x_min, v.x_max), (1.0, 3.0));
        assert_eq!((v.y_min, v.y_max), (0.0, 2.0));
    }
}
