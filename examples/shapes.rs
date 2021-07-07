// Draw the classic triangle to the screen
extern crate quicksilver;

use quicksilver::{
    geom::Vector,
    graphics::{Color, Mesh, ShapeRenderer},
    input::{ButtonState, Key},
    lifecycle::{run, Event, Settings, State, Window},
    lyon::{
        geom::math::*,
        tessellation::{
            basic_shapes::*,
            {FillOptions, StrokeOptions},
        },
    },
    Result,
};

struct ShapesExample {
    shapes: Mesh,
}

impl State for ShapesExample {
    fn new() -> Result<ShapesExample> {

        let stroke_options = StrokeOptions::tolerance(0.01)
            .with_line_width(1.0);

        let mut mesh = Mesh::new();
        let mut renderer = ShapeRenderer::new(&mut mesh, Color::BLACK);

        let result = stroke_circle(
            point(100.0, 100.0),
            10.0,
            &stroke_options,
            &mut renderer,
        ).unwrap();
        eprintln!("result vertices={:?} indices={:?}", result.vertices, result.indices);

        stroke_rounded_rectangle(
            &rect(100.0, 140.0, 100.0, 40.0),
            &BorderRadii {
                top_left: 2.0,
                top_right: 2.0,
                bottom_right: 3.0,
                bottom_left: 3.0,
            },
            &stroke_options,
            &mut renderer,
        ).unwrap();

        let fill_options = FillOptions::tolerance(0.01);

        let result = fill_circle(
            point(160.0, 100.0),
            10.0,
            &fill_options,
            &mut renderer,
        ).unwrap();


        eprintln!("result vertices={:?} indices={:?}", result.vertices, result.indices);

        Ok(ShapesExample {
            shapes: mesh,
        })
    }

    fn event(&mut self, event: &Event, window: &mut Window) -> Result<()> {
        match *event {
            Event::Key(Key::Escape, ButtonState::Pressed) => {
                window.close();
            }
            _ => (),
        }
        Ok(())
    }

    fn draw(&mut self, window: &mut Window) -> Result<()> {
        window.clear(Color::WHITE)?;
        window.mesh().extend(&self.shapes);
        Ok(())
    }
}

fn main() {
    run::<ShapesExample>(
        "Circle Demo - press Space to switch between tessellation methods",
        Vector::new(800, 600),
        Settings {
            multisampling: Some(4),
            ..Settings::default()
        },
    );
}
