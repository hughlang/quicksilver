/// Model object that holds GL instructions and configs.
///
///
use crate::{
    backend::Backend,
    graphics::{GpuTriangle, Vertex},
    lifecycle::Window,
};

/// This is a wrapper object that holds the info necessary for executing a set of GL instructions.
/// This modularity allows Quicksilver to handle multiple GL processing jobs separately with
/// different shader programs and expected outputs.
// #[derive(Clone)]
pub struct Texture {
    /// The ImageData.id value returned by glGenTextures in backend.create_texture.
    pub texture_id: u32,
    /// The id returned by glCreateProgram in backend.link_program.
    pub program_id: u32,
    /// The id returned when glCreateShader in backend.compile_shader for the vertex shader
    pub vertex_id: u32,
    /// The id returned when glCreateShader in backend.compile_shader for the fragment shader
    pub fragment_id: u32,
    /// List of name and field width values
    pub fields: Vec<(String, u32)>,
    /// Identifies the name of the fragment shader variable for the output color
    pub out_color: String,
    /// Identifies the name of the sampler2D variable in the fragment shader
    pub sampler: String,
    /// Function that serializes a Vertex struct into the vec of f32 vals that the GPU shader expects.
    pub serializer: Box<dyn Fn(Vertex) -> Vec<f32> + 'static>,
}

impl Default for Texture {
    fn default() -> Self {
        let default = |_vertex| -> Vec<f32> {
            Vec::new()
        };
        Texture {
            texture_id: 0,
            program_id: 0,
            vertex_id: 0,
            fragment_id: 0,
            fields: Vec::new(),
            out_color: String::default(),
            sampler: String::default(),
            serializer: Box::new(default),
        }
    }
}

impl Texture {

    /// Initialize both shaders using the provided strings which contain OpenGL/WebGL code
    pub fn init_shaders(mut self, vertex_shader: &str, fragment_shader: &str, window: &mut Window) -> Self {
        unsafe {
            let vs = window.backend().compile_shader(vertex_shader, gl::VERTEX_SHADER);
            let fs = window.backend().compile_shader(fragment_shader, gl::FRAGMENT_SHADER);
            if vs.is_ok() && fs.is_ok() {
                let vs = vs.unwrap();
                let fs = fs.unwrap();
                self.vertex_id = vs;
                self.fragment_id = fs;
                let pid = window.backend().link_program(vs, fs);
                if pid.is_ok() {
                    self.program_id = pid.unwrap();
                }
            }
        }
        self
    }
    /// Set the fields that match the shader program inputs
    pub fn with_fields(mut self, fields: &[(&str, u32)], out_color: &str, sampler: &str, window: &mut Window) -> Self {
        self.fields = fields.iter().map(|(s, n)| (s.to_string(), n.clone())).collect();
        self.out_color = out_color.to_string();
        self.sampler = sampler.to_string();
        self
    }

    /// Set the closure function that is used to convert a vertex into a vector of u32 values
    /// that match the data expected by the GL vertex shader
    pub fn set_serializer<C>(&mut self, cb: C)
    where C: Fn(Vertex) -> Vec<f32> + 'static,
    {
        self.serializer = Box::new(cb);
    }
}

/// A temporary object holding the vertices and triangles to be drawn by the backend.
/// The Window instance has a draw_tasks vector and each task is processed during the draw
/// and flush stages
pub struct DrawTask {
    /// The id value matching the Texture registered in backend.textures_map
    pub texture_id: u32,
    /// All the vertices in the task
    pub vertices: Vec<Vertex>,
    /// All the triangles in the task
    pub triangles: Vec<GpuTriangle>,
}

impl DrawTask {
    /// Create DrawTask with the texture_id matching the Texture saved in backend.textures_map
    pub fn new(id: u32) -> Self {
        DrawTask {
            texture_id: id,
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }
}