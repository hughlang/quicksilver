/// Model object that holds GL instructions and configs.
///
///
use crate::{
    graphics::{GpuTriangle, Vertex},
    lifecycle::{run, Settings, State, Window},
};

/// This is a wrapper object that holds the info necessary for executing a set of GL instructions.
/// This modularity allows Quicksilver to handle multiple GL processing jobs separately with
/// different shader programs and expected outputs.
pub struct GLTask {
    /// All the vertices in the task
    pub vertices: Vec<Vertex>,
    /// All the triangles in the task
    pub triangles: Vec<GpuTriangle>,
    /// List of name and field width values
    pub fields: Vec<(String, i32)>,
    /// The id value returned when creating a texture
    pub texture_id: u32,
    /// The texture location
    pub location_id: u32,
    shader_id: u32,
    fragment_id: u32,
    vertex_id: u32,

}

impl Default for GLTask {
    fn default() -> Self {
        GLTask {
            vertices: Vec::new(),
            triangles: Vec::new(),
            fields: Vec::new(),
            texture_id: 0,
            location_id: 0,
            shader_id: 0,
            fragment_id: 0,
            vertex_id: 0,
        }
    }
}

impl GLTask {

    /// Constructor
    pub fn with_shaders(mut self, vertex_shader: &str, fragment_shader: &str, window: &mut Window) -> Self {
        // let vs = window.
        self
    }

    /// Set the fields that match the shader program inputs
    pub fn set_fields(&mut self, fields: &[(&str, i32)]) {

    }
            // let position_string = CString::new("position").expect("No interior null bytes in shader").into_raw();
            // let tex_coord_string = CString::new("tex_coord").expect("No interior null bytes in shader").into_raw();
            // let color_string = CString::new("color").expect("No interior null bytes in shader").into_raw();
            // let tex_string = CString::new("tex").expect("No interior null bytes in shader").into_raw();
            // let use_texture_string = CString::new("uses_texture").expect("No interior null bytes in shader").into_raw();

    fn configure_fields(&self) {
        let spec = &[
            ("left_top", 3),
            ("right_bottom", 2),
            ("tex_left_top", 2),
            ("tex_right_bottom", 2),
            ("color", 4),
        ];
    }

}