//! CPU-side geometry for axis chrome (spines, tick marks): turns an
//! [`Axes2d`]'s already-computed layout into pixel-space rectangles for the
//! chrome pipeline in `chrome.wgsl`.
//!
//! Kept in `rustyplot-render` rather than `rustyplot-core`: it deals in exact
//! pixel geometry for a specific rendering approach (filled quads), whereas
//! core only owns the backend-agnostic layout maths (margins, tick values).

use bytemuck::{Pod, Zeroable};
use rustyplot_core::Axes2d;

/// Colour of spines and tick marks: a dark, neutral grey rather than pure
/// black, matching the restrained "minimal" style (open box, no gridlines).
pub const CHROME_COLOR: [f32; 4] = [0.25, 0.25, 0.25, 1.0];
pub const SPINE_WIDTH: f32 = 1.2;
pub const TICK_LENGTH: f32 = 4.0;
pub const TICK_WIDTH: f32 = 1.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ChromeInstance {
    pub rect: [f32; 4],
    pub color: [f32; 4],
}

impl ChromeInstance {
    fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            rect: [x, y, w, h],
            color: CHROME_COLOR,
        }
    }
}

/// Position of a data value along the plot's local x axis, in pixels from
/// the left edge of `axes.rect`.
fn local_x(axes: &Axes2d, value: f32) -> f32 {
    (value - axes.view.x_min) / axes.view.width() * axes.rect.width
}

/// Position of a data value along the plot's local y axis, in pixels from
/// the top edge of `axes.rect` (data y increases upward, pixels increase downward).
fn local_y(axes: &Axes2d, value: f32) -> f32 {
    (axes.view.y_max - value) / axes.view.height() * axes.rect.height
}

/// Build the spine and tick-mark rectangles for one axes, in figure-absolute
/// pixel coordinates (i.e. already offset by `axes.rect`'s origin).
pub fn build_chrome(axes: &Axes2d) -> Vec<ChromeInstance> {
    let rect = axes.rect;
    let mut instances = Vec::new();

    // Minimal style: only the left and bottom spines (open box), matching
    // matplotlib's "despine" look rather than a fully closed box.
    instances.push(ChromeInstance::new(
        rect.x - SPINE_WIDTH / 2.0,
        rect.y,
        SPINE_WIDTH,
        rect.height,
    ));
    instances.push(ChromeInstance::new(
        rect.x,
        rect.y + rect.height - SPINE_WIDTH / 2.0,
        rect.width,
        SPINE_WIDTH,
    ));

    let (xticks, _) = axes.xticks();
    for v in xticks {
        let x = rect.x + local_x(axes, v);
        if x < rect.x - 0.5 || x > rect.x + rect.width + 0.5 {
            continue;
        }
        instances.push(ChromeInstance::new(
            x - TICK_WIDTH / 2.0,
            rect.y + rect.height,
            TICK_WIDTH,
            TICK_LENGTH,
        ));
    }

    let (yticks, _) = axes.yticks();
    for v in yticks {
        let y = rect.y + local_y(axes, v);
        if y < rect.y - 0.5 || y > rect.y + rect.height + 0.5 {
            continue;
        }
        instances.push(ChromeInstance::new(
            rect.x - TICK_LENGTH,
            y - TICK_WIDTH / 2.0,
            TICK_LENGTH,
            TICK_WIDTH,
        ));
    }

    instances
}

/// Absolute x position of a given x tick, for centring its label.
pub fn xtick_x(axes: &Axes2d, value: f32) -> f32 {
    axes.rect.x + local_x(axes, value)
}

/// Absolute y position of a given y tick, for vertically centring its label.
pub fn ytick_y(axes: &Axes2d, value: f32) -> f32 {
    axes.rect.y + local_y(axes, value)
}

const MARQUEE_BORDER_WIDTH: f32 = 1.0;
const MARQUEE_FILL_COLOR: [f32; 4] = [0.15, 0.45, 0.9, 0.15];
const MARQUEE_BORDER_COLOR: [f32; 4] = [0.15, 0.45, 0.9, 0.9];

/// Box-zoom drag rectangle overlay: a translucent fill plus a thin border,
/// reusing the chrome pipeline (a plain instanced-quad renderer) rather than
/// adding a whole new pipeline for one rectangle.
pub fn build_marquee(x: f32, y: f32, width: f32, height: f32) -> [ChromeInstance; 5] {
    [
        ChromeInstance {
            rect: [x, y, width, height],
            color: MARQUEE_FILL_COLOR,
        },
        // Border, drawn as four thin strips rather than a stroked outline
        // (the chrome pipeline only knows how to fill axis-aligned rects).
        ChromeInstance {
            rect: [x, y, width, MARQUEE_BORDER_WIDTH],
            color: MARQUEE_BORDER_COLOR,
        },
        ChromeInstance {
            rect: [x, y + height - MARQUEE_BORDER_WIDTH, width, MARQUEE_BORDER_WIDTH],
            color: MARQUEE_BORDER_COLOR,
        },
        ChromeInstance {
            rect: [x, y, MARQUEE_BORDER_WIDTH, height],
            color: MARQUEE_BORDER_COLOR,
        },
        ChromeInstance {
            rect: [x + width - MARQUEE_BORDER_WIDTH, y, MARQUEE_BORDER_WIDTH, height],
            color: MARQUEE_BORDER_COLOR,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyplot_core::view::{Rect, View2d};

    fn axes() -> Axes2d {
        let mut a = Axes2d {
            view: View2d::new(0.0, 10.0, 0.0, 10.0),
            ..Default::default()
        };
        a.layout(Rect::new(0.0, 0.0, 400.0, 300.0));
        a
    }

    #[test]
    fn chrome_includes_two_spines_and_some_ticks() {
        let a = axes();
        let instances = build_chrome(&a);
        // At least the two spines plus a handful of tick marks.
        assert!(instances.len() > 2);
    }

    #[test]
    fn ticks_stay_within_the_axes_rect() {
        let a = axes();
        for inst in build_chrome(&a) {
            let [x, y, w, h] = inst.rect;
            assert!(x >= a.rect.x - TICK_LENGTH - 1.0);
            assert!(y + h <= a.rect.y + a.rect.height + TICK_LENGTH + 1.0);
            assert!(w > 0.0 && h > 0.0);
        }
    }
}
