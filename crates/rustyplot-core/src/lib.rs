//! Backend-agnostic core of rustyplot.
//!
//! Holds the scene description, view maths and interaction logic. It knows nothing
//! about GPUs, windows or Python, which is what allows a second [`Backend`]
//! (an SVG writer, say) to be added without touching this crate.

pub mod axes;
pub mod interaction;
pub mod scene;
pub mod ticks;
pub mod view;

pub use axes::Axes2d;
pub use interaction::{Interaction, PickHit};
pub use scene::{Line, LineSeries, LineStyle, Marker, MarkerStyle, Scene, ScatterSeries, SeriesError};
pub use ticks::{format_ticks, nice_ticks};
pub use view::{Rect, View2d, Viewport};

/// A renderer capable of turning a [`Scene`] into pixels (or vectors).
pub trait Backend {
    type Error;

    fn resize(&mut self, width: u32, height: u32);
    fn draw(&mut self, scene: &Scene) -> Result<(), Self::Error>;
}
