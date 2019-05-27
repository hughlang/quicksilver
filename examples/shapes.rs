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
            StrokeOptions,
        },
    },
    Result,
};

struct ShapesExample {
    stroke_circle: Mesh,
    draw_filled: bool,
}

impl State for ShapesExample {
    fn new() -> Result<ShapesExample> {

        let stroke_options = StrokeOptions::tolerance(0.01)
            .with_line_width(1.0);

        let mesh = {
            let mut mesh = Mesh::new();
            let mut renderer = ShapeRenderer::new(&mut mesh, Color::BLACK);
            // renderer.set_transform(Transform::scale((3, 3)));
            let result = stroke_circle(
                point(100.0, 100.0),
                10.0,
                &stroke_options,
                &mut renderer,
            ).unwrap();
            eprintln!("result vertices={:?} indices={:?}", result.vertices, result.indices);

            mesh
        };

        Ok(ShapesExample {
            stroke_circle: mesh,
            draw_filled: true,
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
        window.mesh().extend(&self.stroke_circle);
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
