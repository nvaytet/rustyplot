//! An `Axes2d` is one viewport within a figure: its own data view, its own
//! draw list, its own decorations. The figure is a grid of these from the
//! start (see `AGENTS.md`), so adding a second subplot later is just a bigger
//! grid, not a rewrite of the plot area.

use crate::scene::{LineSeries, ScatterSeries};
use crate::ticks::nice_ticks;
use crate::view::{Rect, View2d};

/// Target number of ticks per axis; `nice_ticks` may return a couple more or
/// fewer to keep round numbers.
pub const TARGET_TICK_COUNT: usize = 5;

// Heuristic text metrics used to size margins before a single glyph is drawn.
// `rustyplot-core` has no font, so this is a deliberate approximation (see
// AGENTS.md's dependency-free constraint on this crate); the render crate
// draws at these same sizes so the estimate and the reality stay close.
pub const TICK_FONT_SIZE: f32 = 11.0;
pub const AXIS_LABEL_FONT_SIZE: f32 = 13.0;
pub const TITLE_FONT_SIZE: f32 = 15.0;
const CHAR_WIDTH_FACTOR: f32 = 0.62;
const LINE_HEIGHT_FACTOR: f32 = 1.3;
const TICK_LENGTH: f32 = 4.0;
const TICK_LABEL_GAP: f32 = 3.0;
const FIGURE_PADDING: f32 = 8.0;

fn text_width(text: &str, font_size: f32) -> f32 {
    text.chars().count() as f32 * font_size * CHAR_WIDTH_FACTOR
}

fn line_height(font_size: f32) -> f32 {
    font_size * LINE_HEIGHT_FACTOR
}

/// One subplot: a data view, the artists drawn into it, and its decorations.
#[derive(Clone, Debug, PartialEq)]
pub struct Axes2d {
    pub view: View2d,
    pub scatter: Vec<ScatterSeries>,
    pub lines: Vec<LineSeries>,
    pub title: String,
    pub xlabel: String,
    pub ylabel: String,
    /// Pixel rect of the plot area (inside the spines), set by [`Layout::compute`].
    pub rect: Rect,
}

impl Default for Axes2d {
    fn default() -> Self {
        Self {
            view: View2d::default(),
            scatter: Vec::new(),
            lines: Vec::new(),
            title: String::new(),
            xlabel: String::new(),
            ylabel: String::new(),
            rect: Rect::default(),
        }
    }
}

impl Axes2d {
    pub fn point_count(&self) -> usize {
        self.scatter.iter().map(ScatterSeries::len).sum::<usize>()
            + self.lines.iter().map(LineSeries::len).sum::<usize>()
    }

    /// Fit the view to all series in this axes.
    pub fn autoscale(&mut self, margin: f32) {
        let mut x: Vec<f32> = Vec::new();
        let mut y: Vec<f32> = Vec::new();
        for s in &self.scatter {
            x.extend_from_slice(&s.x);
            y.extend_from_slice(&s.y);
        }
        for l in &self.lines {
            x.extend_from_slice(&l.x);
            y.extend_from_slice(&l.y);
        }
        if !x.is_empty() {
            self.view = View2d::from_points(&x, &y, margin);
        }
    }

    /// X tick positions and labels for the current view.
    pub fn xticks(&self) -> (Vec<f32>, Vec<String>) {
        axis_ticks(self.view.x_min, self.view.x_max)
    }

    /// Y tick positions and labels for the current view.
    pub fn yticks(&self) -> (Vec<f32>, Vec<String>) {
        axis_ticks(self.view.y_min, self.view.y_max)
    }

    /// Margins this axes needs around its plot area, given its own decorations.
    ///
    /// `ylabel` runs vertically alongside the y axis, as matplotlib's does.
    /// `rustyplot-render` achieves this by shaping it into a small offscreen
    /// texture (glyphon has no rotated-text primitive) and drawing that
    /// texture as a quad rotated 90°; here we only need its *thickness*
    /// (one line height) to reserve room for it in the left margin, since
    /// rotation swaps width and height.
    fn margins(&self) -> (f32, f32, f32, f32) {
        let (_, xlabels) = self.xticks();
        let (_, ylabels) = self.yticks();
        let max_ylabel_width = ylabels
            .iter()
            .map(|s| text_width(s, TICK_FONT_SIZE))
            .fold(0.0f32, f32::max);
        let max_xlabel_width = xlabels
            .iter()
            .map(|s| text_width(s, TICK_FONT_SIZE))
            .fold(0.0f32, f32::max);

        let left = FIGURE_PADDING
            + if self.ylabel.is_empty() {
                0.0
            } else {
                line_height(AXIS_LABEL_FONT_SIZE) + TICK_LABEL_GAP
            }
            + max_ylabel_width
            + TICK_LABEL_GAP
            + TICK_LENGTH;

        let bottom = FIGURE_PADDING
            + if self.xlabel.is_empty() {
                0.0
            } else {
                line_height(AXIS_LABEL_FONT_SIZE) + TICK_LABEL_GAP
            }
            + line_height(TICK_FONT_SIZE)
            + TICK_LABEL_GAP
            + TICK_LENGTH;

        let top = FIGURE_PADDING
            + if self.title.is_empty() {
                0.0
            } else {
                line_height(TITLE_FONT_SIZE)
            };

        // No right spine in the default (minimal) style, but the last x tick
        // label can overhang the plot area; give it room not to be clipped.
        let right = FIGURE_PADDING + max_xlabel_width / 2.0;

        (left, right, top, bottom)
    }

    /// Compute `self.rect` from `cell`, the pixel rect this axes owns within
    /// the figure grid, reserving space for ticks and labels.
    pub fn layout(&mut self, cell: Rect) {
        let (left, right, top, bottom) = self.margins();
        self.rect = cell.shrink(left, right, top, bottom);
    }
}

fn axis_ticks(min: f32, max: f32) -> (Vec<f32>, Vec<String>) {
    let ticks = nice_ticks(min, max, TARGET_TICK_COUNT);
    let labels = if ticks.len() >= 2 {
        crate::ticks::format_ticks(&ticks, ticks[1] - ticks[0])
    } else {
        crate::ticks::format_ticks(&ticks, (max - min).max(f32::EPSILON))
    };
    (ticks, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_shrinks_cell_by_margins() {
        let mut axes = Axes2d {
            view: View2d::new(0.0, 100.0, 0.0, 100.0),
            ..Default::default()
        };
        let cell = Rect::new(0.0, 0.0, 400.0, 300.0);
        axes.layout(cell);
        assert!(axes.rect.x > 0.0);
        assert!(axes.rect.y > 0.0);
        assert!(axes.rect.width < cell.width);
        assert!(axes.rect.height < cell.height);
    }

    #[test]
    fn labels_and_title_grow_their_margins() {
        let mut plain = Axes2d {
            view: View2d::new(0.0, 100.0, 0.0, 100.0),
            ..Default::default()
        };
        let mut decorated = Axes2d {
            view: View2d::new(0.0, 100.0, 0.0, 100.0),
            title: "Title".into(),
            xlabel: "x axis".into(),
            ylabel: "y axis".into(),
            ..Default::default()
        };
        let cell = Rect::new(0.0, 0.0, 400.0, 300.0);
        plain.layout(cell);
        decorated.layout(cell);
        // ylabel grows the left margin (it runs vertically, rotated,
        // alongside the y axis); title grows the top margin.
        assert!(decorated.rect.x > plain.rect.x);
        assert!(decorated.rect.y > plain.rect.y);
        assert!(decorated.rect.width < plain.rect.width);
        assert!(decorated.rect.height < plain.rect.height);
    }

    #[test]
    fn xticks_and_yticks_match_view() {
        let axes = Axes2d {
            view: View2d::new(0.0, 10.0, -5.0, 5.0),
            ..Default::default()
        };
        let (xt, _) = axes.xticks();
        let (yt, _) = axes.yticks();
        assert_eq!(xt, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(yt, vec![-4.0, -2.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn point_count_and_autoscale_include_line_series() {
        let mut axes = Axes2d {
            lines: vec![LineSeries {
                x: vec![0.0, 10.0],
                y: vec![0.0, 4.0],
                color: [0.0; 4],
                line: Some(crate::scene::Line {
                    width: 2.0,
                    style: crate::scene::LineStyle::Solid,
                }),
                marker: None,
            }],
            ..Default::default()
        };
        assert_eq!(axes.point_count(), 2);
        axes.autoscale(0.0);
        assert_eq!((axes.view.x_min, axes.view.x_max), (0.0, 10.0));
        assert_eq!((axes.view.y_min, axes.view.y_max), (0.0, 4.0));
    }
}
