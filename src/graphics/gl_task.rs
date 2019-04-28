/// Model object that holds GL instructions and configs.
///
///
use crate::{
    Result,
    backend::Backend,
    geom::{Rectangle, Vector},
    graphics::{GpuTriangle, PixelFormat, Vertex},
    lifecycle::Window,
};

use gl::types::*;

/// This is a wrapper object that holds the info necessary for executing a set of GL instructions.
/// This modularity allows Quicksilver to handle multiple GL processing jobs separately with
/// different shader programs and expected outputs.
pub struct GLTask {
    /// All the vertices in the task
    pub vertices: Vec<Vertex>,
    /// All the triangles in the task
    pub triangles: Vec<GpuTriangle>,
    /// List of name and field width values
    pub fields: Vec<(String, u32)>,
    /// The id value returned when creating a texture
    pub texture_id: u32,
    /// The texture location
    pub location_id: u32,
    program_id: u32,
    fragment_shader_id: u32,
    vertex_shader_id: u32,

}

impl Default for GLTask {
    fn default() -> Self {
        GLTask {
            vertices: Vec::new(),
            triangles: Vec::new(),
            fields: Vec::new(),
            texture_id: 0,
            location_id: 0,
            program_id: 0,
            fragment_shader_id: 0,
            vertex_shader_id: 0,
        }
    }
}

impl GLTask {

    /// Constructor
    pub fn init_shaders(mut self, vertex_shader: &str, fragment_shader: &str, window: &mut Window) -> Self {
        unsafe {
            let vs = window.backend().compile_shader(vertex_shader, gl::VERTEX_SHADER);
            let fs = window.backend().compile_shader(fragment_shader, gl::FRAGMENT_SHADER);
            if vs.is_ok() && fs.is_ok() {
                let vs = vs.unwrap();
                let fs = fs.unwrap();
                self.vertex_shader_id = vs;
                self.fragment_shader_id = fs;
                let pid = window.backend().link_program(vs, fs);
                if pid.is_ok() {
                    self.program_id = pid.unwrap();
                }
            }
        }
        self
    }
    /// Set the fields that match the shader program inputs
    pub fn with_fields(mut self, fields: &[(&str, u32)], out_color: &str, window: &mut Window) -> Self {
        self.fields = fields.iter().map(|(s, n)| (s.to_string(), n.clone())).collect();
        self
    }

    /// Method to create a texture in the GPU with specified width, height, and pixel_format
    pub fn create_texture(&mut self, width: u32, height: u32, pixel_format: PixelFormat, window: &mut Window) -> Result<u32> {
        let data = window.create_texture(&[], width, height, pixel_format).unwrap();
        self.texture_id = data.id;
        Ok(self.texture_id)
    }

    /// Passthru method to update the texture
    pub fn update_texture(&mut self, data: &[u8], rect: &Rectangle, format: PixelFormat, window: &mut Window) {

    }

}